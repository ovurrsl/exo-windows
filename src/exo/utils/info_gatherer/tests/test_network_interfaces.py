import socket
from types import SimpleNamespace

import psutil
import pytest

from exo.utils.info_gatherer import system_info


@pytest.mark.parametrize(
    ("iface_name", "expected"),
    [
        ("Ethernet", "ethernet"),
        ("Wi-Fi", "wifi"),
        ("vEthernet (WSL)", "unknown"),
        ("Hyper-V Virtual Ethernet Adapter", "unknown"),
        ("Teredo Tunneling Pseudo-Interface", "unknown"),
        ("Loopback Pseudo-Interface 1", "unknown"),
    ],
)
def test_guess_interface_type_ignores_virtual_adapters(
    iface_name: str,
    expected: str,
) -> None:
    assert system_info._guess_interface_type(iface_name) == expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_get_network_interfaces_filters_windows_virtual_and_down_nics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_addrs = {
        "Ethernet": [
            SimpleNamespace(family=socket.AF_INET, address="192.168.1.101"),
            SimpleNamespace(
                family=socket.AF_INET6, address="fe80::1aa8:7728:316e:3a66"
            ),
        ],
        "Wi-Fi": [
            SimpleNamespace(family=socket.AF_INET, address="169.254.73.24"),
        ],
        "vEthernet (WSL)": [
            SimpleNamespace(family=socket.AF_INET, address="172.24.80.1"),
        ],
        "Teredo Tunneling Pseudo-Interface": [
            SimpleNamespace(
                family=socket.AF_INET6, address="2001:0:14c9:d804:386b:168a:a0b9:774c"
            ),
        ],
        "Loopback Pseudo-Interface 1": [
            SimpleNamespace(family=socket.AF_INET, address="127.0.0.1"),
            SimpleNamespace(family=socket.AF_INET6, address="::1"),
        ],
    }
    fake_stats = {
        "Ethernet": SimpleNamespace(isup=True),
        "Wi-Fi": SimpleNamespace(isup=False),
        "vEthernet (WSL)": SimpleNamespace(isup=True),
        "Teredo Tunneling Pseudo-Interface": SimpleNamespace(isup=True),
        "Loopback Pseudo-Interface 1": SimpleNamespace(isup=True),
    }

    async def no_macos_setup() -> dict[str, object]:
        return {}

    monkeypatch.setattr(system_info.sys, "platform", "win32")
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: fake_addrs)
    monkeypatch.setattr(psutil, "net_if_stats", lambda: fake_stats)
    monkeypatch.setattr(
        system_info, "_get_interface_types_from_networksetup", no_macos_setup
    )

    interfaces = await system_info.get_network_interfaces()
    assert [(nic.name, nic.ip_address, nic.interface_type) for nic in interfaces] == [
        ("Ethernet", "192.168.1.101", "ethernet"),
        ("Ethernet", "fe80::1aa8:7728:316e:3a66", "ethernet"),
    ]
