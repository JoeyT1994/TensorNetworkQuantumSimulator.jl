# Head-to-head at MATCHED bond dimension: CTMRG `:cut` vs CTMRG `:cycle` vs boundary MPS.
#
# The 5x5 Ising benchmark (examples/ctm_ising5x5_benchmark.jl) is ONE physical state and flatters
# `:cycle`. This is the breadth check: three families that stress different things, all measured
# against the same `alg="exact"` ground truth, with wall time alongside accuracy.
#
#   A. random-bond classical Ising   single layer, "flat" -- no ket/bra doubling, signed couplings
#   B. random PEPS norm              double layer, random signed -- the adversarial case
#   C. short-time dynamics           double layer, near-product -- what people actually run
#
# B is NOT a proxy for C. A random signed PEPS is near the worst case for every method here; a
# lightly-evolved state is near a product state and easy for all of them. The interesting regime is
# C at late times, which this script deliberately does not reach -- raise `NL`/`MAXDIM` to push it,
# but keep `alg="exact"` affordable (5x5 at maxdim=4 already does not finish).
#
# BOUNDARY MPS is called as `expect(psi, obs; alg="boundarymps", mps_bond_dimension=chi)`.
#
# ⚠️ THE TWO bMPS ENTRY POINTS DO NOT AGREE, and the reason is a DEFAULT, not an algorithm:
#
#     BoundaryMPSCache(psi, chi)                  gauge_state = false   (boundarympscache.jl)
#     expect(psi, obs; alg="boundarymps")         gauge_state = true    (expect.jl)
#
# They are otherwise the same computation -- `partition_by` is NOT the cause (the cache default is
# already `row`, exactly what `expect` picks here), and setting `gauge_state=true` on an explicitly
# built cache reproduces the `expect` path to every digit. Worth aligning: a user cross-checking
# `expect` against a hand-built cache currently gets different numbers with nothing to explain it.
#
# The accuracy consequence is NOT systematic. Over 3 lattices x 3 chi x 5 seeds the error ratio
# (gauged/ungauged) scatters from 0.14 to 1.22 with no chi trend and hurt-counts of 0-4 of 5 -- i.e.
# a coin flip. An earlier single-seed reading of "12x worse at chi=2" did not survive more seeds.
#
# TIMING. Every measurement is run TWICE and the SECOND is reported. The contraction-sequence cache
# is keyed on tensor SHAPE, so a first call at a new shape pays netcon plus JIT (5-14x). This trap
# has produced wrong speed claims in this project more than once.
#
# Run: julia --project=. --startup-file=no examples/ctm_projector_survey.jl        (~2-4 min)

using TensorNetworkQuantumSimulator
using ITensors
using Random
using Printf
using Dictionaries: Dictionary
const TNQS = TensorNetworkQuantumSimulator

const CHIS = (2, 4, 8)
const METHODS = (:cut, :cycle, :bmps)
const LABEL = Dict(:cut => ":cut", :cycle => ":cycle", :bmps => "bMPS")

# Run `f` twice, return (value, time_of_second_call).
function timed2(f)
    f()
    t = @elapsed v = f()
    return v, t
end

function header(title, metric)
    @printf("\n%s\n  metric: %s\n  %-5s", title, metric, "chi")
    for m in METHODS
        @printf(" | %-10s %-6s", LABEL[m], "s")
    end
    @printf("\n  %s\n", "-"^(5 + 20 * length(METHODS)))
end

function row(χ, r)
    best = argmin(m -> r[m][1], collect(METHODS))
    @printf("  %-5d", χ)
    for m in METHODS
        @printf(" | %-9.3e%s %-6.2f", r[m][1], m == best ? "*" : " ", r[m][2])
    end
    println()
end

# --------------------------------------------- single layer: compare ln Z

function survey_partitionfunction(title, tn)
    lnZ = log(abs(real(contract(tn; alg = "exact"))))
    header(title, @sprintf("|ln Z - exact|,  exact = %.10f", lnZ))
    for χ in CHIS
        r = Dict{Symbol, Tuple{Float64, Float64}}()
        for p in (:cut, :cycle)
            v, t = timed2(() -> cvm_freenergy(update(CTMEnvironmentCache(tn, χ; projector = p))))
            r[p] = (abs(v - lnZ), t)
        end
        v, t = timed2(() -> log(abs(real(contract(tn; alg = "boundarymps", mps_bond_dimension = χ)))))
        r[:bmps] = (abs(v - lnZ), t)
        row(χ, r)
    end
end

# ------------------------------------ double layer: compare a local <Z>

function survey_state(title, ψ, v)
    ex = real(expect(ψ, ("Z", [v]); alg = "exact"))
    header(title, @sprintf("|<Z> - exact| at %s,  exact = %.10f", string(v), ex))
    obs = ("Z", [v])
    for χ in CHIS
        r = Dict{Symbol, Tuple{Float64, Float64}}()
        for p in (:cut, :cycle)
            z, t = timed2(() -> real(expect(update(CTMEnvironmentCache(ψ, χ; projector = p)), obs)))
            r[p] = (abs(z - ex), t)
        end
        z, t = timed2(() -> real(expect(ψ, obs; alg = "boundarymps", mps_bond_dimension = χ)))
        r[:bmps] = (abs(z - ex), t)
        row(χ, r)
    end
end

# ------------------------------------------------------------- families

# A. Random-bond classical Ising: single layer, SIGNED couplings (N(0,1)), so spin-glass-like
#    rather than a ferromagnet. No ket/bra doubling anywhere.
function family_A()
    Random.seed!(1)
    g = named_grid((6, 6))
    Js = Dictionary(edges(g), [randn() for _ in edges(g)])
    survey_partitionfunction("A. random-bond Ising 6x6, beta=0.4 (single layer, signed)",
                             ising_partitionfunction(g, 0.4; Js))
end

# B. Random PEPS norm: double layer, random signed amplitudes. The hardest case for `:cycle` and
#    its standing weakness at D=3 (see docs/ctmrg_status.md).
function family_B()
    Random.seed!(1)
    g = named_grid((5, 5))
    ψ = random_tensornetworkstate(Float64, g, siteinds("S=1/2", g); bond_dimension = 3)
    survey_state("B. random PEPS norm 5x5 D=3 (double layer, signed)", ψ, (3, 3))
end

# C. Short-time dynamics: Trotterised transverse-field Ising on |0...0>. Double layer but
#    STRUCTURED and near-product -- the regime a real simulation lives in.
#    4x4, not 5x5: at 5x5 with maxdim=4 the exact contraction does not finish, and without exact
#    ground truth the whole comparison is worthless.
function family_C(; nl = 3, maxdim = 4)
    g = named_grid((4, 4))
    dt, hx, hz, J = 0.25, 1.0, 0.8, 0.5
    layer = []
    append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
    append!(layer, ("Rz", [v], 2 * hz * dt) for v in vertices(g))
    for colored_edges in edge_color(g, 4)
        append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
    end
    ψ_bpc = BeliefPropagationCache(tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2"))
    for _ in 1:nl
        ψ_bpc, _ = apply_gates(layer, ψ_bpc;
                               apply_kwargs = (; maxdim, cutoff = 1.0e-12, normalize_tensors = false),
                               verbose = false)
    end
    survey_state("C. TFIM dynamics 4x4, $nl layers, maxdim=$maxdim (double layer, structured)",
                 network(ψ_bpc), (2, 2))
end

function main()
    @printf("CTMRG :cut vs :cycle vs boundary MPS, matched bond dimension\n")
    @printf("'*' marks the best method on that row. Each timing is the SECOND of two calls.\n")
    @printf("NOTE chi is not an equal-cost knob across methods: CTM keeps chi per interface AND\n")
    @printf("carries corner tensors, bMPS keeps chi on the MPS bonds. Read the trend per method.\n")
    family_A()
    family_B()
    family_C()
    println()
end

main()
