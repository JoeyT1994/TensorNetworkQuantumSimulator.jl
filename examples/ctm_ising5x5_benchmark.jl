# Head-to-head against the collaborator's finite CTMRG on their own 5x5 Ising PEPS.
#
# Needs `peps5x5.bin`, produced with `python examples/export_peps.py 5x5`.
#     python examples/export_peps.py 5x5
# (override the location with the CTM_ISING5X5_DIR environment variable). Compares the CUT projector
# against the CYCLE projector
# (`projector = :cut` vs `:cycle`) on `ln<psi|psi>` and on a single-site <X>.
#
# THE EXPORT TRAP, recorded because it cost a full detour. The npz PICKLES JAX ARRAYS, so
# `np.load(..., allow_pickle=True)` runs jax on unpickle -- and without x64 enabled every tensor
# comes back float32 (0.24756516516 against the true 0.24756516631). `configure_jax()` must run
# BEFORE `np.load`. Avoiding jax in the export script does not dodge this; it triggers it. A
# float32 export shifts ln Z by 7.5e-8 and <X> by 2.1e-9 -- small enough to look like a real
# finding about the collaborator's engine, and it fooled me into claiming exactly that. The doc's
# reference values are correct:
#     ln<psi|psi> = -6.217866847854575    <X> at their (2,2) = 0.916900598128483
# This script recomputes both with `alg="exact"` and ASSERTS against those, so a degraded export
# is caught here rather than showing up as a spurious projector result.
#
# Run: julia --project=. --startup-file=no examples/ctm_ising5x5_benchmark.jl

using TensorNetworkQuantumSimulator
using ITensors
using Dictionaries: Dictionary, set!
using NamedGraphs: NamedEdge
using Printf
const TNQS = TensorNetworkQuantumSimulator

const BIN = joinpath(get(ENV, "CTM_ISING5X5_DIR", @__DIR__), "peps5x5.bin")
const NX, NY = 5, 5

# Raw legs are (p, l, r, d, u) with l -> (x-1,y), r -> (x+1,y), d -> (y-1), u -> (y+1); every
# boundary leg has dimension 1, so the shape is fully determined by position (verified against the
# file). The buffer is C-ordered, so reshape reversed and permute back.
rawshape(x, y) = (2, x == 1 ? 1 : 3, x == NX ? 1 : 3, y == 1 ? 1 : 3, y == NY ? 1 : 3)

function load_peps()
    isfile(BIN) || error("""
        missing $BIN -- generate it with:
            python $(joinpath(@__DIR__, "export_peps.py")) 5x5
        or point CTM_ISING5X5_DIR at the directory holding peps5x5.bin.""")
    data = open(BIN, "r") do f
        read!(f, Vector{Float64}(undef, filesize(BIN) ÷ 8))
    end
    off = 0
    arrs = Dict{Tuple{Int, Int}, Array{Float64, 5}}()
    for x in 1:NX, y in 1:NY
        sh = rawshape(x, y)
        n = prod(sh)
        v = @view data[(off + 1):(off + n)]
        arrs[(x, y)] = permutedims(reshape(collect(v), reverse(sh)), (5, 4, 3, 2, 1))
        off += n
    end
    @assert off == length(data) "buffer not fully consumed: $off of $(length(data))"
    return arrs
end

function build_state(arrs)
    g = named_grid((NX, NY))
    s = siteinds("S=1/2", g)
    link = Dict{Any, Index}()
    for e in edges(g)
        link[e] = Index(3, "Link")
        link[reverse(e)] = link[e]
    end
    tensors = Dictionary{Tuple{Int, Int}, ITensor}()
    for x in 1:NX, y in 1:NY
        a = arrs[(x, y)]                       # (p, l, r, d, u)
        v = (x, y)
        # drop the dimension-1 boundary legs -- they are not graph edges
        nbrs = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        keep = [i for i in 1:4 if size(a, i + 1) > 1]
        @assert all(i -> nbrs[i] in vertices(g), keep) "kept a leg with no neighbour at $v"
        @assert Set(keep) == Set(i for i in 1:4 if nbrs[i] in vertices(g)) "leg/neighbour mismatch at $v"
        idx = ITensors.Index[only(s[v])]
        for i in keep
            push!(idx, link[NamedEdge(v => nbrs[i])])
        end
        set!(tensors, v, ITensor(reshape(a, (2, [size(a, i + 1) for i in keep]...)), idx))
    end
    return TNQS.TensorNetworkState(TNQS.TensorNetwork(tensors, g), s)
end

function main()
    ψ = build_state(load_peps())
    lnN = log(abs(real(norm_sqr(ψ; alg = "exact"))))
    site = (3, 3)                              # their (2,2), 0-indexed
    exX = real(expect(ψ, ("X", [site]); alg = "exact"))

    # Guard the transfer, not just the algorithm: a float32-degraded export lands ~7.5e-8 off on
    # lnZ and ~2.1e-9 on <X>, which is far too small to notice in the table below.
    @printf("\n5x5 Ising PEPS D=3, ITensors exact contraction\n")
    @printf("  ln<psi|psi> = %.16g   (reference -6.217866847854575, delta %.2e)\n",
            lnN, abs(lnN - (-6.217866847854575)))
    @printf("  <X> at %-6s = %.16g   (reference  0.916900598128483, delta %.2e)\n",
            string(site), exX, abs(exX - 0.916900598128483))
    @assert abs(lnN - (-6.217866847854575)) < 1e-12 "transfer is degraded: check configure_jax() ran before np.load"
    @assert abs(exX - 0.916900598128483) < 1e-12 "transfer is degraded: check configure_jax() ran before np.load"

    # Their engine's <X> error at matched chi, measured from their own primitives against this
    # same reference (contract_Z11 with an explicit (X.t, t) pair). NOT the doc's Python column:
    # that quotes 8.149e-14 at chi=32, where direct measurement gives 1.330e-12.
    theirs = Dict(4 => 5.218e-5, 9 => 5.132e-8, 16 => 9.279e-10, 32 => 1.330e-12)

    # `marginal_inconsistency` is the only ln Z-free quality measure here, and it is the point of
    # `:cycle`: it should sit at machine precision at every χ, where `:cut` does not.
    @printf("\n%-4s %-11s %-11s %-11s %-11s %-11s %-11s %-11s\n", "chi",
            "lnN cut", "lnN cyc", "<X> cut", "<X> cyc", "<X> THEIRS", "marg cut", "marg cyc")
    for χ in (4, 9, 16, 32)
        r = Dict{Symbol, NTuple{3, Float64}}()
        for proj in (:cut, :cycle)
            c = update(CTMEnvironmentCache(ψ, χ; projector = proj))
            r[proj] = (abs(cvm_freenergy(c) - lnN),
                       abs(real(expect(c, ("X", [site]))) - exX),
                       marginal_inconsistency(c))
        end
        @printf("%-4d %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e\n", χ,
                r[:cut][1], r[:cycle][1], r[:cut][2], r[:cycle][2], theirs[χ],
                r[:cut][3], r[:cycle][3])
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
