# Local PEPS data

Place collaborator `.npz`/`.npy` fixtures under `examples/data/peps/`. The directory is ignored by
Git because these are research inputs, not package source. Convert the Ising fixtures with:

```text
python examples/export_peps.py 5x5
python examples/export_peps.py 9x9
```

The generated raw buffers are also ignored and are read by the corresponding Julia benchmarks.
