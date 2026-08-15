"""Build the native compressed Sylvester solver extension."""

from pathlib import Path

import numpy as np
import jax
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated" / "sylvester_solvers"


def _dedupe_flags(flags):
    """Return compiler or linker flags with duplicates removed."""
    seen = set()
    deduped = []
    for flag in flags:
        if flag in seen:
            continue
        seen.add(flag)
        deduped.append(flag)
    return deduped


class BuildExt(build_ext):
    """Build while avoiding duplicate Conda rpath flags on macOS."""

    def build_extensions(self):
        """Deduplicate compiler and linker command lists before compilation."""
        for attr in ("compiler_so", "linker_so", "linker_exe"):
            flags = getattr(self.compiler, attr, None)
            if flags is not None:
                setattr(self.compiler, attr, _dedupe_flags(flags))
        super().build_extensions()


extensions = [
    Extension(
        "linalg.sylvester_solvers._sylvester_solvers",
        [str(HERE / "_sylvester_solvers.pyx"), str(HERE / "ffi.cc")],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=[np.get_include(), jax.ffi.include_dir()],
        extra_compile_args=["-std=c++17"],
        language="c++",
    )
]


if __name__ == "__main__":
    setup(
        name="sylvester_solvers",
        cmdclass={"build_ext": BuildExt},
        ext_modules=cythonize(
            extensions,
            build_dir=str(GENERATED),
            include_path=[str(HERE.parents[1])],
            compiler_directives={"language_level": 3},
        ),
    )
