"""Build hooks that compile the native Zig core and bundle it into the wheel.

zpds is not a CPython extension — the Python bindings talk to a prebuilt shared
library through cffi's dlopen (ABI) mode. So the packaging job is: compile
``libzpds`` with Zig, drop it next to the ``zpds`` package, and tag the wheel as
platform-specific (but Python-ABI-agnostic: ``py3-none-<platform>``).

Zig is located as either the ``zig`` executable on PATH or, failing that, the
``ziglang`` PyPI package (``python -m ziglang ...``), which is what CI installs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

ROOT = Path(__file__).resolve().parent
PKG_DIR = ROOT / "python" / "zpds"


def _lib_filename() -> str:
    if sys.platform == "darwin":
        return "libzpds.dylib"
    if sys.platform == "win32":
        return "zpds.dll"
    return "libzpds.so"


def _zig_command() -> list[str]:
    if shutil.which("zig"):
        return ["zig"]
    # Fall back to the `ziglang` wheel (installed via cibuildwheel before-all).
    return [sys.executable, "-m", "ziglang"]


def _zig_target() -> str | None:
    """Best-effort cross-compilation target for the current wheel build.

    Honors an explicit ``ZPDS_ZIG_TARGET``; otherwise infers from the platform
    hint cibuildwheel/pip set (``_PYTHON_HOST_PLATFORM``, ``ARCHFLAGS``) so
    macOS arm64/x86_64 wheels compile for the right arch.
    """
    explicit = os.environ.get("ZPDS_ZIG_TARGET")
    if explicit:
        return explicit

    hints = (
        os.environ.get("_PYTHON_HOST_PLATFORM", "")
        + " "
        + os.environ.get("ARCHFLAGS", "")
    ).lower()
    os_part = {"darwin": "macos", "linux": "linux", "win32": "windows"}.get(sys.platform)
    if os_part is None:
        return None
    if "arm64" in hints or "aarch64" in hints:
        return f"aarch64-{os_part}"
    if "x86_64" in hints:
        return f"x86_64-{os_part}"
    return None


def _build_native() -> Path:
    cmd = _zig_command() + ["build", "-Doptimize=ReleaseSafe"]
    target = _zig_target()
    if target:
        cmd.append(f"-Dtarget={target}")
    print("zpds: building native core:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)
    lib = ROOT / "zig-out" / "lib" / _lib_filename()
    if not lib.is_file():
        raise SystemExit(f"zpds: expected native library at {lib} but it is missing")
    return lib


class BuildWithZig(build_py):
    """Compile the Zig core and copy the shared library into the package."""

    def run(self) -> None:
        lib = _build_native()
        # Copy into the source tree (editable installs / sdist-in-place) ...
        shutil.copy2(lib, PKG_DIR / lib.name)
        super().run()
        # ... and into the staged build output that becomes the wheel.
        staged = Path(self.build_lib) / "zpds"
        staged.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lib, staged / lib.name)


class BinaryDistribution(Distribution):
    """Mark the distribution as platform-specific.

    zpds ships a prebuilt shared library rather than a CPython ``Extension``,
    so setuptools would otherwise treat every file as *purelib* and stage the
    library under ``<name>.data/purelib/`` in the wheel — which ``auditwheel``
    rejects ("shared library in purelib folder"). Reporting ext modules routes
    the package (and the bundled ``libzpds``) to the platlib root instead.
    """

    def has_ext_modules(self) -> bool:  # noqa: D401 - simple override
        return True


def _wheel_cmdclass() -> dict:
    """A bdist_wheel that marks the wheel impure and Python-ABI-agnostic."""
    try:  # setuptools >= 70 vendors bdist_wheel
        from setuptools.command.bdist_wheel import bdist_wheel as _base
    except ImportError:  # older: comes from the `wheel` package
        from wheel.bdist_wheel import bdist_wheel as _base

    class BDistWheel(_base):
        def finalize_options(self) -> None:
            super().finalize_options()
            self.root_is_pure = False  # platform-specific wheel

        def get_tag(self):
            _, _, plat = super().get_tag()
            return "py3", "none", plat  # works on any Python 3

    return {"bdist_wheel": BDistWheel}


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildWithZig, **_wheel_cmdclass()},
)
