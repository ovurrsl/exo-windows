from collections.abc import Generator, Iterator
from typing import Any

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import ToolCallItem
from exo.shared.types.chunks import (
    ErrorChunk,
    GenerationChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import ModelId
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.models.registry import architecture_for_model_class
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import (
    ToolParser,
    coerce_tool_calls_to_schemas,
    tool_names,
)


def count_reasoning_tokens(
    responses: Generator[GenerationResponse | ToolCallResponse | None],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    """Count tokens with is_thinking=True and patch the total into Usage on the final response."""
    reasoning_tokens = 0
    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, GenerationResponse) and response.is_thinking:
            reasoning_tokens += 1
        if response.usage is not None and reasoning_tokens > 0:
            response = response.model_copy(
                update={
                    "usage": response.usage.model_copy(
                        update={
                            "completion_tokens_details": response.usage.completion_tokens_details.model_copy(
                                update={"reasoning_tokens": reasoning_tokens}
                            )
                        }
                    )
                }
            )
        yield response


def apply_all_parsers(
    receiver: Generator[GenerationResponse | None],
    prompt: str,
    tool_parser: ToolParser | None,
    tokenizer: TokenizerWrapper,
    model_type: type[Model],
    model_id: ModelId,
    tools: list[dict[str, Any]] | None,
) -> Iterator[GenerationChunk | None]:
    generator = receiver

    normalized_id = model_id.normalize().lower()
    architecture = architecture_for_model_class(model_type)
    output_parser = None if architecture is None else architecture.output_parser
    if (
        output_parser is not None
        and output_parser.applies_to_model_id(normalized_id)
        and output_parser.applies_to_prompt(prompt)
    ):
        if output_parser.after_thinking_parser and tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )
        if output_parser.parse_with_tool_names is not None and tools is not None:
            generator = output_parser.parse_with_tool_names(
                generator, tool_names(tools)
            )
        else:
            generator = output_parser.parse(generator)
        if output_parser.coerces_to_tool_schemas and tools:
            generator = coerce_tool_call_responses(generator, tools)
    else:
        if tokenizer.has_thinking:
            generator = parse_thinking_models(
                generator,
                tokenizer.think_start,
                tokenizer.think_end,
                starts_in_thinking=detect_thinking_prompt_suffix(prompt, tokenizer),
            )

        if tool_parser:
            generator = parse_tool_calls(generator, tool_parser, tools)

    generator = count_reasoning_tokens(generator)

    return map(lambda r: map_responses_to_chunks(r, model_id), generator)


def coerce_tool_call_responses(
    responses: Generator[GenerationResponse | ToolCallResponse | None],
    tools: list[dict[str, Any]],
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    for response in responses:
        if isinstance(response, ToolCallResponse):
            response = response.model_copy(
                update={
                    "tool_calls": coerce_tool_calls_to_schemas(
                        response.tool_calls, tools
                    )
                }
            )
        yield response


def map_responses_to_chunks(
    response: GenerationResponse | ToolCallResponse | None, model_id: ModelId
) -> GenerationChunk | None:
    match response:
        case None:
            return None
        case GenerationResponse():
            if response.finish_reason == "error":
                return ErrorChunk(
                    error_message=response.text,
                    model=model_id,
                )
            else:
                finish_reason = response.finish_reason
                assert finish_reason not in (
                    "error",
                    "tool_calls",
                    "function_call",
                )
                return TokenChunk(
                    model=model_id,
                    text=response.text,
                    token_id=response.token,
                    usage=response.usage,
                    finish_reason=finish_reason,
                    stats=response.stats,
                    logprob=response.logprob,
                    top_logprobs=response.top_logprobs,
                    is_thinking=response.is_thinking,
                )
        case ToolCallResponse():
            return ToolCallChunk(
                tool_calls=response.tool_calls,
                model=model_id,
                usage=response.usage,
                stats=response.stats,
            )


def _longest_tag_prefix_suffix(text: str, tag: str) -> int:
    """Return length of the longest non-empty suffix of ``text`` matching a strict prefix of ``tag``."""
    max_len = min(len(text), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


def _emit_prefix_chunks(
    pending_buffer: list[GenerationResponse],
    template_response: GenerationResponse,
    prefix_len: int,
    extra_from_template: str,
    is_thinking: bool,
) -> Iterator[GenerationResponse]:
    """Emit ``prefix_len`` chars spanning ``pending_buffer`` + ``extra_from_template``."""
    remaining = prefix_len
    for buffered in pending_buffer:
        if remaining <= 0:
            break
        if len(buffered.text) <= remaining:
            yield buffered.model_copy(update={"is_thinking": is_thinking})
            remaining -= len(buffered.text)
        else:
            yield buffered.model_copy(
                update={"text": buffered.text[:remaining], "is_thinking": is_thinking}
            )
            remaining = 0
    if remaining > 0:
        yield template_response.model_copy(
            update={
                "text": extra_from_template[:remaining],
                "is_thinking": is_thinking,
            }
        )


def parse_thinking_models(
    responses: Generator[GenerationResponse | None],
    think_start: str | None,
    think_end: str | None,
    starts_in_thinking: bool = True,
) -> Generator[GenerationResponse | None]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens (including when BPE merges tag boundaries with
    adjacent punctuation or whitespace), sets is_thinking on all others, and
    always yields tokens with finish_reason to avoid hanging the chunk stream.
    """
    is_thinking = starts_in_thinking
    pending_buffer: list[GenerationResponse] = []

    def drain_pending(_is_thinking: bool) -> Iterator[GenerationResponse]:
        for buffered in pending_buffer:
            yield buffered.model_copy(update={"is_thinking": _is_thinking})
        pending_buffer.clear()

    for response in responses:
        if response is None:
            yield None
            continue

        if response.finish_reason is not None:
            yield from drain_pending(is_thinking)
            yield response.model_copy(update={"is_thinking": False})
            continue

        remaining_text = response.text
        if not remaining_text:
            if not pending_buffer:
                yield response.model_copy(update={"is_thinking": is_thinking})
            continue

        while remaining_text:
            active_tag = think_end if is_thinking else think_start
            if not active_tag:
                yield from drain_pending(is_thinking)
                yield response.model_copy(
                    update={"text": remaining_text, "is_thinking": is_thinking}
                )
                break

            accumulated = "".join(item.text for item in pending_buffer)
            combined = accumulated + remaining_text
            tag_idx = combined.find(active_tag)

            if tag_idx != -1:
                if tag_idx > 0:
                    yield from _emit_prefix_chunks(
                        pending_buffer,
                        response,
                        tag_idx,
                        remaining_text,
                        is_thinking,
                    )
                pending_buffer.clear()
                is_thinking = not is_thinking
                remaining_text = combined[tag_idx + len(active_tag) :]
                continue

            keep = _longest_tag_prefix_suffix(combined, active_tag)
            if keep == len(combined):
                pending_buffer.append(
                    response
                    if remaining_text == response.text
                    else response.model_copy(update={"text": remaining_text})
                )
                break

            emit_len = len(combined) - keep
            yield from _emit_prefix_chunks(
                pending_buffer,
                response,
                emit_len,
                remaining_text,
                is_thinking,
            )
            pending_buffer.clear()
            if keep > 0:
                pending_buffer.append(
                    response.model_copy(update={"text": combined[emit_len:]})
                )
            break


def parse_tool_calls(
    responses: Generator[GenerationResponse | None],
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
) -> Generator[GenerationResponse | ToolCallResponse | None]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    accumulated_tool_calls: list[ToolCallItem] = []

    for response in responses:
        if response is None:
            yield None
            continue

        if not in_tool_call and response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if (
            not in_tool_call
            and accumulated_tool_calls
            and (response.stats is not None or response.finish_reason is not None)
        ):
            yield ToolCallResponse(
                tool_calls=accumulated_tool_calls,
                usage=response.usage,
                stats=response.stats,
            )
            accumulated_tool_calls.clear()
            continue

        if not in_tool_call:
            yield response
            continue

        tool_call_text_parts.append(response.text)
        if response.text.endswith(tool_parser.end_parsing):
            # parse the actual tool calls from the tool call text
            combined = "".join(tool_call_text_parts)
            parsed = tool_parser.parse(combined.strip(), tools=tools)
            logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
            in_tool_call = False
            tool_call_text_parts = []

            if parsed is None:
                # What the model wrote isn't a tool call we can read: give it back as text, as
                # other OpenAI-compatible servers do, rather than failing the request
                logger.warning(f"tool call parsing failed for text {combined}")
                yield response.model_copy(update={"text": combined, "token": 0})
                continue

            accumulated_tool_calls.extend(parsed)
            if accumulated_tool_calls and (
                response.finish_reason is not None or response.stats is not None
            ):
                yield ToolCallResponse(
                    tool_calls=accumulated_tool_calls,
                    usage=response.usage,
                    stats=response.stats,
                )
                accumulated_tool_calls.clear()
            continue

        if response.finish_reason is not None:
            logger.info(
                "tool call parsing interrupted, yield partial tool call as text"
            )
            response = response.model_copy(
                update={"text": "".join(tool_call_text_parts), "token": 0}
            )
            yield response

    if not accumulated_tool_calls:
        logger.warning("Tool calls should have all been emitted but were not")
