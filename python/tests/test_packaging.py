"""End-to-end packaging test: build a wheel and inspect it.

Verifies that ``setup.py`` compiles the Zig core and bundles the shared library
into a platform-specific, Python-ABI-agnostic wheel. Skipped when the build
toolchain (Zig, the ``build`` frontend) is unavailable, so the rest of the
suite still runs in minimal environments.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _has_zig() -> bool:
    if shutil.which("zig"):
        return True
    try:
        import ziglang  # noqa: F401

        return True
    except ImportError:
        return False


# --no-isolation reuses the current environment, so these must be importable.
pytest.importorskip("build", reason="the `build` frontend is not installed")
pytest.importorskip("setuptools", reason="setuptools not installed")
pytest.importorskip("wheel", reason="wheel not installed")
pytestmark = pytest.mark.skipif(not _has_zig(), reason="Zig toolchain not available")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("wheelhouse")
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", str(out)],
        cwd=ROOT,
    )
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def test_wheel_is_platform_specific_but_abi_agnostic(wheel: Path):
    # Tag is py3-none-<platform>: any Python 3, but not a pure "any" wheel.
    name = wheel.name
    assert "-py3-none-" in name
    assert "none-any" not in name


def test_wheel_bundles_native_library(wheel: Path):
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    libs = [n for n in names if n.rsplit("/", 1)[-1].startswith("libzpds") or n.endswith("zpds.dll")]
    assert libs, f"no native library bundled in wheel; contents: {names}"


def test_wheel_contains_python_package(wheel: Path):
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    assert any(n.endswith("zpds/__init__.py") for n in names)
    assert any(n.endswith("zpds/_native.py") for n in names)
