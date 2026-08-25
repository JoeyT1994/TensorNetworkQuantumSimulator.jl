"""Convert the local pickled-JAX Ising PEPS fixtures to Julia's raw buffer format.

Usage:
    python examples/export_peps.py 5x5
    python examples/export_peps.py 9x9-D2
    python examples/export_peps.py 9x9-D3

The source ``.npz`` files live under ``examples/data/peps`` and are intentionally
ignored by Git. They are JAX-array pickles, but only their NumPy payload is needed here.
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np


HERE = Path(__file__).resolve().parent
CASES = {
    "5x5": (5, 3, "data_ising_5x5/isingZZX_5x5_D3_g3.04438.npz", "peps5x5.bin"),
    "9x9-D2": (9, 2, "data_ising_9x9/IsingZZX_9x9_D2_g3.04438_gs.npz", "peps9x9_D2.bin"),
    "9x9-D3": (9, 3, "data_ising_9x9/isingZZX_9x9_D3_g3.04438.npz", "peps9x9_D3.bin"),
}


def site_shape(x: int, y: int, length: int, bond_dimension: int) -> tuple[int, ...]:
    edge = length - 1
    return (2, 1 if x == 0 else bond_dimension,
            1 if x == edge else bond_dimension,
            1 if y == 0 else bond_dimension,
            1 if y == edge else bond_dimension)


def _reconstruct_jax_array(reconstructor, args, state, _metadata):
    """Rebuild the NumPy payload stored by JAX without importing JAX."""

    array = reconstructor(*args)
    array.__setstate__(state)
    return array


class _JaxArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "jax._src.array" and name == "_reconstruct_array":
            return _reconstruct_jax_array
        return super().find_class(module, name)


def load_arrays(path: Path):
    with path.open("rb") as stream:
        return _JaxArrayUnpickler(stream).load()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    length, bond_dimension, relative_source, output_name = CASES[args.case]
    source = HERE / "data" / "peps" / relative_source
    output = HERE / output_name

    raw = load_arrays(source)
    assert (len(raw), len(raw[0])) == (length, length)
    with output.open("wb") as stream:
        for x in range(length):
            for y in range(length):
                array = np.asarray(raw[x][y])
                expected = site_shape(x, y, length, bond_dimension)
                assert array.dtype == np.float64 and array.shape == expected, (
                    x, y, array.dtype, array.shape, expected)
                stream.write(np.ascontiguousarray(array).tobytes(order="C"))

    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
