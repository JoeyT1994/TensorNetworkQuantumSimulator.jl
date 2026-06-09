using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge
Random.seed!(1234)
using NPZ

using TensorNetworkQuantumSimulator
using Graphs: edges, src, dst, vertices
using LinearAlgebra: Diagonal, transpose, exp

# C <- conj(u) C transpose(u), u = identity except a 2x2 block w on modes (a,b)
function _apply_two_mode!(C, a, b, w)
    blk = [a, b]
    C[blk, :] .= conj(w) * C[blk, :]
    C[:, blk] .= C[:, blk] * transpose(w)
    return C
end
# u = identity except scalar phase ph = exp(-i theta) on mode a
function _apply_one_mode!(C, a, ph)
    C[a, :] .*= conj(ph)
    C[:, a] .*= ph
    return C
end

# Exact (truncation-free) reference for the free-fermion circuit in free_fermions.jl.
# Returns the trajectory of  C[v1,v2] + C[v2,v1]  (== expect(obs1)+expect(obs2)),
# at t = 0, dt, 2dt, ..., nsteps*dt.  Reproduces the EXACT same Trotter circuit
# (gate order, edge_color grouping), so the only difference vs the TN run is bond
# truncation.  C[i,j] = <c_i^dag c_j>.
function exact_hoppings(μ, g, dt, t, nsteps, v1, v2, mod)
    vs = collect(vertices(g))
    idx = Dict(v => k for (k, v) in enumerate(vs))

    # initial product state: occupied where sum(v) is even
    occ = [sum(v) % mod == 0 ? 0.0 : 1.0 for v in vs]
    C0 = complex(Matrix(Diagonal(occ)))

    # per-gate single-particle pieces, matching tofermionicitensor conventions
    w_hop  = exp(-im * (t * dt) * ComplexF64[0 1; 1 0])         # RHop: exp(-i θ (E_ij+E_ji)), θ=t*dt
    ph_num = Dict(v => exp(-im * (isodd(sum(v)) ? -μ * dt : μ * dt)) for v in vs)  # RN: ±μ*dt

    hop_order = collect(Iterators.flatten(edge_color(g, 3)))    # same order as the example

    i1, i2 = idx[v1], idx[v2]
    obs(C) = real(C[i1, i2] + C[i2, i1])

    traj = zeros(Float64, nsteps + 1)
    traj[1] = obs(C0)
    C = copy(C0)
    for n in 1:nsteps
        for v in vs;         _apply_one_mode!(C, idx[v], ph_num[v]); end
        for e in hop_order;  _apply_two_mode!(C, idx[src(e)], idx[dst(e)], w_hop); end
        for v in vs;         _apply_one_mode!(C, idx[v], ph_num[v]); end
        traj[n + 1] = obs(C)
    end
    return (0:nsteps) .* dt, traj
end

function main_fermions(χ, μ, lattice)

    g = lattice == "Hexagonal" ? named_hexagonal_lattice_graph(4,4; periodic = false) : named_comb_tree((8,6))
    s = siteinds("fermion", g)
    mod = 4
    ψ = fermionic_tensornetworkstate(Float64, v -> sum(v) % mod == 0 ? "Emp" : "Occ", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    println("Real time Evo in a staggered field")
    dt = 0.01

    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = nothing)
    single_site_gates = [isodd(sum(v)) ? ("RN", v, -μ*dt) : ("RN", v, μ*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nsteps = 100
    t_update =0
    t_bp = 0

    v = first(center(g))
    vn = first(neighbors(g, v))
    @assert v ∈ vertices(g) && vn ∈ vertices(g)
    obs1, obs2 = [(["Cdag", "C"], [v, vn])],[(["Cdag", "C"], [vn, v])]
    cidag_cjs = ComplexF64[only(expect(ψ_bpc, obs1)) + only(expect(ψ_bpc, obs2))]
    times, cidag_cjs_exact = exact_hoppings(μ, g, dt, t, nsteps, v, vn, mod)
    bp_times =Float64[0]
    gate_app_times = Float64[0]
    times = [i*dt for i in 0:nsteps]
    for i in 1:nsteps
        t1 = time()
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        t2 = time()

        ψ_bpc = update(ψ_bpc; tolerance = nothing, niters =50)
        t3 = time()

        t_update += (t2-t1)
        t_bp += (t3-t2)

        cidag_cj = only(expect(ψ_bpc, obs1)) + only(expect(ψ_bpc, obs2))
        cidag_cjexact = only(cidag_cjs_exact[i + 1])
        push!(cidag_cjs, cidag_cj)
        push!(bp_times, t3-t2 )
        push!(gate_app_times, t2-t1)

        if i % 1 == 0
            println("Time is $(i*dt)")
            println("This Time step took $(t3-t1) secs")
            println("Current BD is $(maxvirtualdim(ψ_bpc))")
            println("BP Measured hopping is $cidag_cj")
            println("Exact hopping is $(cidag_cjexact)")

            println("Absolute Error is $(abs(cidag_cjexact - cidag_cj))")
        end

    end

    Rs = [1,2,4,8,16,32]
    cidag_cj_bmps = []
    for R in Rs
        ψ_bmps = update(BoundaryMPSCache(network(ψ_bpc), R; partition_by = "col"))
        cidag_cj = only(expect(ψ_bmps, obs1)) + only(expect(ψ_bmps, obs2))
        push!(cidag_cj_bmps, cidag_cj)

        println("Absolute Error on BMPS value at $R is $(abs(cidag_cj - only(cidag_cjs_exact[nsteps + 1])))")
    end

    @show Rs
    @show cidag_cj_bmps
    @show cidag_cjs_exact[nsteps + 1]

    #npzwrite("/Users/jtindall/Files/Data/Fermions/SpinlessFreeFermionQuenchMu$(μ)BD$(χ)$(lattice).npz", cidag_cjs_exact = cidag_cjs_exact, cidag_cjs_bp = cidag_cjs,
    #    times= times,gate_app_times = gate_app_times, bp_times = bp_times)
end

χs = [8]
mus =[1.0]
lattices = ["Hexagonal"]
for lattice in lattices
    for χ in χs
        for mu in mus
            main_fermions(χ, mu, lattice)
        end
    end
end

