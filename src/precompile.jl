# Precompile workload: a fresh session paid ~85 s of JIT for a small dense workflow and ~68 s for a
# graded one (measured 2026-09-10, package already precompiled). Exercising the main paths here at
# tiny sizes moves that compilation into the package image (measured after: 0.4 s dense, 8.8 s
# graded, +3 s load). Kept minimal on purpose — every line adds to the package's own precompile
# time — and wrapped so a failure in the workload never breaks loading the package.
using PrecompileTools: @setup_workload, @compile_workload
using Logging: Logging

function _precompile_workload(g)
    # dense
    s = siteinds("S=1/2", g)
    ψ = random_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)
    expect(ψ, ("Z", [(1, 1)]); alg = "bp")
    expect(ψ, ("Z", [(1, 1)]); alg = "exact")
    ψ2, _ = apply_gates(Any[("Rzz", ((1, 1), (1, 2)), 0.1), ("Rx", [(1, 1)], 0.2)], ψ; apply_kwargs = (; maxdim = 4))
    expect(update(CTMEnvironmentCache(ψ2, 4); maxiter = 1), ("Z", [(1, 1)]))
    expect(update(CTMEnvironmentCache(ψ2, 4; projector = :cycle); maxiter = 1, convergence = :marginal), ("Z", [(1, 1)]))
    expect(ψ2, ("Z", [(1, 1)]); alg = "boundarymps", mps_bond_dimension = 4)
    # graded (fermions; Z2 shares the code path)
    sf = siteinds("Fermion", g; symmetry = "fZ2")
    ψf = tensornetworkstate(ComplexF64, v -> v == (1, 1) ? "Occ" : "Emp", g, sf)
    ψf, _ = apply_gates(Any[("F_hop", ((1, 1), (1, 2)), 0.3)], ψf; apply_kwargs = (; maxdim = 4))
    expect(ψf, ("N", [(1, 1)]); alg = "bp")
    expect(ψf, ("N", [(1, 1)]); alg = "exact")
    expect(update(CTMEnvironmentCache(ψf, 4); maxiter = 1), ("N", [(1, 1)]))
    expect(ψf, ("N", [(1, 1)]); alg = "boundarymps", mps_bond_dimension = 4)
    return nothing
end

# Development switch: `TNQS_SKIP_PRECOMPILE_WORKLOAD=1 julia ...` skips the workload (package
# precompile back to ~20 s) for edit–run loops; the PrecompileTools `precompile_workload`
# preference does the same persistently.
@setup_workload begin
    g = named_grid((2, 2))
    haskey(ENV, "TNQS_SKIP_PRECOMPILE_WORKLOAD") || @compile_workload begin
        try
            # one-sweep CTM runs warn about not certifying; keep the precompile log clean
            Logging.with_logger(() -> _precompile_workload(g), Logging.NullLogger())
        catch err
            @warn "precompile workload skipped" exception = (err, catch_backtrace())
        end
    end
end
