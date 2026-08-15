"""Build the local f2py SLICOT extension with release optimization.

The default source is extracted from the bundled pristine SLICOT 5.9.1 release
archive under ``linalg/periodic_schur/SLICOT``. Pass
``--source /path/to/SLICOT-Reference`` to test another source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


SLICOT_VERSION = "5.9.1"
HERE = Path(__file__).resolve().parent
BUNDLED_SLICOT_ARCHIVE = (
    HERE / "SLICOT" / f"SLICOT-Reference-{SLICOT_VERSION}.tar.gz"
)
BUNDLED_SLICOT_SHA256 = (
    "37b0c0fc1800454f8d7553a004a5bb6fac9e042fe2b592ce1eec92045ce9b7a1"
)
SLICOT_ROUTINES = (
    "MA01BD",
    "MA01BZ",
    "MB03AB",
    "MB03AF",
    "MB03BA",
    "MB03BB",
    "MB03BC",
    "MB03BD",
    "MB03BF",
    "MB03BZ",
    "MB03KA",
    "MB03KB",
    "MB03KC",
    "MB03KD",
    "MB03KE",
    "MB03VD",
    "MB03VY",
    "MB03WD",
    "MB03WX",
    "MB04PY",
)
FORTRAN_FLAGS = (
    "-O3",
    "-fPIC",
    "-frecursive",
    "-fallow-argument-mismatch",
    "-mcpu=native",
    "-mtune=native",
)


def _extract_bundled_source(build_dir):
    """Verify and extract the bundled SLICOT release archive."""
    with BUNDLED_SLICOT_ARCHIVE.open("rb") as archive_file:
        digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    if digest != BUNDLED_SLICOT_SHA256:
        raise RuntimeError(
            f"unexpected SLICOT archive SHA-256: {digest}; "
            f"expected {BUNDLED_SLICOT_SHA256}"
        )

    with tarfile.open(BUNDLED_SLICOT_ARCHIVE, "r:gz") as archive:
        archive.extractall(build_dir, filter="data")
    return build_dir / f"SLICOT-Reference-{SLICOT_VERSION}"


def _fortran_compiler():
    """Find the active environment's gfortran executable."""
    for variable in ("FC", "F77", "F90"):
        compiler = os.environ.get(variable)
        if compiler and shutil.which(compiler):
            return Path(shutil.which(compiler))

    prefix_bin = Path(sys.prefix) / "bin"
    candidates = sorted(prefix_bin.glob("*-gfortran"))
    candidates.extend((prefix_bin / "gfortran", Path("gfortran")))
    for candidate in candidates:
        compiler = shutil.which(str(candidate))
        if compiler:
            return Path(compiler)
    raise RuntimeError("gfortran was not found in the active Python environment or PATH")


def _source_files(source_root):
    """Return the exact SLICOT source closure exposed by the local pyf file."""
    source_dir = source_root / "src"
    files = [source_dir / f"{routine}.f" for routine in SLICOT_ROUTINES]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing SLICOT sources: {missing}")
    return files


def _link_environment(compiler):
    """Return a release-build environment with the intended BLAS linkage."""
    environment = os.environ.copy()
    for variable in ("FC", "F77", "F90"):
        environment[variable] = str(compiler)
    environment["FFLAGS"] = " ".join(FORTRAN_FLAGS)
    if sys.platform == "darwin":
        link_flags = environment.get("LDFLAGS", "").split()
        link_flags.extend(("-framework", "Accelerate"))
        environment["LDFLAGS"] = " ".join(link_flags)
    return environment


def _build_command(source_root):
    """Return the f2py command for the optimized SLICOT source closure."""
    pyf = Path(__file__).with_name("slicot_periodic.pyf")
    command = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        str(pyf),
        *(str(path) for path in _source_files(source_root)),
        f"--f77flags={' '.join(FORTRAN_FLAGS)}",
        "--backend",
        "meson",
    ]
    if sys.platform != "darwin":
        command.extend((f"-L{Path(sys.prefix) / 'lib'}", "-llapack", "-lblas"))
    return command


def _build_static_library(source_root, compiler, build_dir):
    """Compile the required SLICOT closure and return a static archive path."""
    object_dir = build_dir / "slicot_static"
    object_dir.mkdir()
    objects = []
    for source in _source_files(source_root):
        object_path = object_dir / f"{source.stem}.o"
        subprocess.run(
            [str(compiler), *FORTRAN_FLAGS, "-c", str(source), "-o", str(object_path)],
            check=True,
        )
        objects.append(object_path)
    archive = object_dir / "libslicot_periodic.a"
    subprocess.run(["ar", "rcs", str(archive), *(str(path) for path in objects)], check=True)
    return archive


def _verify_extension(extension, build_dir):
    """Smoke-test the extension surface and its macOS Accelerate linkage."""
    code = (
        "import _slicot_periodic as module; "
        "required = ('mb03vd', 'mb03vy', 'mb03wd', 'mb03wx', "
        "'mb03kd', 'mb03bd', 'mb03bz'); "
        "assert all(hasattr(module, name) for name in required)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=build_dir, check=True)
    if sys.platform == "darwin":
        linked = subprocess.run(
            ["otool", "-L", str(extension)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "Accelerate.framework" not in linked:
            raise RuntimeError("SLICOT extension is not linked directly to Accelerate")


def build_slicot(source_root=None, output_dir=None):
    """Build and install both Python and static-link SLICOT artifacts."""
    output_dir = Path(output_dir) if output_dir else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler = _fortran_compiler()

    with tempfile.TemporaryDirectory(prefix="build_slicot_") as temporary:
        build_dir = Path(temporary)
        source_root = (
            Path(source_root).resolve()
            if source_root
            else _extract_bundled_source(build_dir)
        )
        command = _build_command(source_root)
        print(f"Fortran compiler: {compiler}")
        print(f"Fortran flags: {' '.join(FORTRAN_FLAGS)}")
        subprocess.run(command, cwd=build_dir, env=_link_environment(compiler), check=True)
        static_library = _build_static_library(source_root, compiler, build_dir)

        extensions = list(build_dir.glob("_slicot_periodic*.so"))
        if len(extensions) != 1:
            raise RuntimeError(f"expected one built extension, found {extensions}")
        extension = extensions[0]
        _verify_extension(extension, build_dir)

        destination = output_dir / extension.name
        temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(extension, temporary_destination)
        os.replace(temporary_destination, destination)
        library_dir = output_dir / "generated" / "build_slicot"
        library_dir.mkdir(parents=True, exist_ok=True)
        library_destination = library_dir / static_library.name
        temporary_library = library_destination.with_suffix(".a.tmp")
        shutil.copy2(static_library, temporary_library)
        os.replace(temporary_library, library_destination)
    print(f"Installed optimized SLICOT extension: {destination}")
    print(f"Installed static SLICOT library: {library_destination}")
    return destination


def _parse_args():
    """Parse command-line arguments for the local build helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="alternate SLICOT-Reference root; defaults to the bundled v5.9.1 archive",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="extension destination (defaults to the periodic_schur package)",
    )
    return parser.parse_args()


def main():
    """Build SLICOT from the command line."""
    args = _parse_args()
    build_slicot(source_root=args.source, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
