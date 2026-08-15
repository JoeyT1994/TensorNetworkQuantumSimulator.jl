# Bundled SLICOT source archive

`SLICOT-Reference-5.9.1.tar.gz` is the complete, unmodified archive from the
upstream `v5.9.1` release:

<https://github.com/SLICOT/SLICOT-Reference/releases/tag/v5.9.1>

The downloaded release archive had SHA-256:

```text
37b0c0fc1800454f8d7553a004a5bb6fac9e042fe2b592ce1eec92045ce9b7a1
```

The upstream license is retained inside the archive at
`SLICOT-Reference-5.9.1/LICENSE`.

`../build_slicot.py` verifies this checksum and extracts the archive into its
temporary build directory. Production currently compiles the explicit
periodic-Schur closure listed in `SLICOT_ROUTINES`; the complete archive is
bundled so routine coverage and dependency choices can be audited without
another download.
