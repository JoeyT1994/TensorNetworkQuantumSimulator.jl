"""Build the unified periodic-Schur Cython extension."""

import jax
import numpy as np
from Cython.Build import cythonize
from pathlib import Path
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated" / "periodic_schur_cython"
SLICOT_LIBRARY = HERE / "generated" / "build_slicot" / "libslicot_periodic.a"
LINK_ARGS = ["-framework", "Accelerate"] if sys.platform == "darwin" else []
LIBRARIES = [] if sys.platform == "darwin" else ["lapack", "blas"]


def _dedupe_flags(flags):
    """Return compiler/linker flags with duplicate entries removed."""
    seen = set()
    deduped = []
    for flag in flags:
        if flag in seen:
            continue
        seen.add(flag)
        deduped.append(flag)
    return deduped


class BuildExt(build_ext):
    """Build extension while avoiding duplicate conda rpath flags on macOS."""

    def build_extensions(self):
        """Deduplicate compiler and linker command lists before compilation."""
        for attr in ("compiler_so", "linker_so", "linker_exe"):
            flags = getattr(self.compiler, attr, None)
            if flags is not None:
                setattr(self.compiler, attr, _dedupe_flags(flags))
        super().build_extensions()


extensions = [
    Extension(
        "linalg.periodic_schur._periodic_schur",
        [str(HERE / "periodic_schur.pyx"), str(HERE / "ffi.cc")],
        depends=[str(SLICOT_LIBRARY)],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_EXTERN_C", 'extern "C"'),
        ],
        include_dirs=[str(HERE), np.get_include(), jax.ffi.include_dir()],
        extra_compile_args=["-std=c++17"],
        extra_objects=[str(SLICOT_LIBRARY)],
        extra_link_args=LINK_ARGS,
        language="c++",
        libraries=LIBRARIES,
    )
]


if __name__ == "__main__":
    setup(
        name="periodic_schur_cy",
        cmdclass={"build_ext": BuildExt},
        ext_modules=cythonize(
            extensions,
            build_dir=str(GENERATED),
            compiler_directives={"language_level": 3},
        ),
    )
