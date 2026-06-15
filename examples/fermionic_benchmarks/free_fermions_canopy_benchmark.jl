using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, add_edge, add_edges, add_vertex
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge, NamedGraphs
Random.seed!(1234)

using Graphs: edges, src, dst, vertices
using LinearAlgebra: Diagonal, transpose, exp

"""
    hexagonal_lattice(m, n; periodic=(false, false)) -> Vector{UndirectedEdge{NTuple{3,Int}}}

Edges of a honeycomb (hexagonal) lattice of `m × n` unit cells, each carrying two sublattice sites.
Vertices are labeled `(i, j, s)` with `i ∈ 1:m`, `j ∈ 1:n`, sublattice `s ∈ {1, 2}`.
Each `A = (i, j, 1)` bonds to `B`-sites `(i, j, 2)`, `(i+1, j, 2)`, `(i, j+1, 2)`, giving coordination 3 in the bulk.

`periodic[1]` / `periodic[2]` wrap the first / second axis (plane / cylinder / torus).
A periodic axis must have size `≥ 2` to avoid duplicated bonds.
"""
function canopy_hexagonal_lattice(m::Int, n::Int; periodic::NTuple{2, Bool} = (false, false))
    (m ≥ 1 && n ≥ 1) || throw(ArgumentError("hexagonal_lattice: dimensions must be ≥ 1"))
    #_check_periodic(periodic, (m, n), (2, 2), "hexagonal_lattice")
    g = NamedGraph()
    es = NamedEdge[]
    for i in 1:m, j in 1:n
        a = (i, j, 1)
        g = add_vertex(g, (i, j, 1))
        g = add_vertex(g, (i, j, 2))
        push!(es, NamedEdge(a => (i, j, 2)))
        if i < m
            push!(es, NamedEdge(a => (i + 1, j, 2)))
        elseif periodic[1]
            push!(es, NamedEdge(a => (1, j, 2)))
        end
        if j < n
            push!(es, NamedEdge(a => (i, j + 1, 2)))
        elseif periodic[2]
            push!(es, NamedEdge(a => (i, 1, 2)))
        end
    end
    g = add_edges(g, es)
    return g
end

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

# Write a wide CSV (one row per Trotter step) from a vector of NamedTuples, using the
# field names of the first row as the header. Dependency-free (no DataFrames/CSV needed),
# so it mirrors the columns produced by the collaborator's `run_timings.jl`.
function write_wide_csv(path, rows)
    isempty(rows) && return path
    ks = keys(rows[1])
    open(path, "w") do io
        println(io, join(string.(ks), ","))
        for r in rows
            println(io, join((string(getfield(r, k)) for k in ks), ","))
        end
    end
    return path
end

function main_fermions(χ, μ, lattice; nsteps = 25, bp_iters = 30,
        outdir = joinpath(@__DIR__, "data"), prefix = "tnqs")

    g = lattice == "Hexagonal" ? canopy_hexagonal_lattice(4,6) : named_comb_tree((8,6))
    s = siteinds("fermion", g)
    mod = 4
    ψ = fermionic_tensornetworkstate(ComplexF64, v -> sum(v) % mod == 0 ? "Emp" : "Occ", g, s)

    nt = Threads.nthreads()
    nsites = nv(g)
    dt = 0.01

    # `@timed` records both wall-clock and heap bytes per phase, matching run_timings.jl.
    # Row `step=0` is the initial BP convergence (gate columns 0.0); run BP for a fixed
    # `bp_iters` sweeps (tol=nothing) so the per-step work is constant and comparable.
    rows = NamedTuple[]
    r0 = @timed update(BeliefPropagationCache(ψ); tolerance = nothing, maxiter = bp_iters)
    ψ_bpc = r0.value
    rescale!(ψ_bpc)
    push!(
        rows, (;
            chi = χ, nthreads = nt, nsites = nsites, dt = dt, step = 0,
            single1 = 0.0, hop = 0.0, single2 = 0.0, bp = r0.time,
            single1_bytes = 0, hop_bytes = 0, single2_bytes = 0, bp_bytes = r0.bytes,
            maxdim = maxvirtualdim(ψ_bpc),
        )
    )

    println("Real time Evo in a staggered field on a lattice with $(nv(g)) vertices")

    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = nothing)
    single_site_gates = [isodd(v[3]) ? ("RN", v, -μ*dt) : ("RN", v, μ*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

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
        r1 = @timed apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = r1.value
        rh = @timed apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = rh.value
        r2 = @timed apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = r2.value

        rb = @timed update(ψ_bpc; tolerance = nothing, maxiter = bp_iters)
        ψ_bpc = rb.value

        t_update += (r1.time + rh.time + r2.time)
        t_bp += rb.time

        push!(
            rows, (;
                chi = χ, nthreads = nt, nsites = nsites, dt = dt, step = i,
                single1 = r1.time, hop = rh.time, single2 = r2.time, bp = rb.time,
                single1_bytes = r1.bytes, hop_bytes = rh.bytes, single2_bytes = r2.bytes, bp_bytes = rb.bytes,
                maxdim = maxvirtualdim(ψ_bpc),
            )
        )

        cidag_cj = only(expect(ψ_bpc, obs1)) + only(expect(ψ_bpc, obs2))
        cidag_cjexact = only(cidag_cjs_exact[i + 1])
        push!(cidag_cjs, cidag_cj)
        push!(bp_times, rb.time)
        push!(gate_app_times, r1.time + rh.time + r2.time)

        if i % 1 == 0
            println("Time is $(i*dt)")
            #println("This Time steps BP update took $(rb.time) secs")
            #println("This Time steps Gate app took $(r1.time + rh.time + r2.time) secs")
            #println("Current BD is $(maxvirtualdim(ψ_bpc))")
            #println("BP Measured hopping is $cidag_cj")
            #println("Exact hopping is $(cidag_cjexact)")

            println("Absolute Error is $(abs(cidag_cjexact - cidag_cj))")
        end

    end

    #npzwrite("/Users/jtindall/Files/Data/Fermions/SpinlessFreeFermionQuenchMu$(μ)BD$(χ)$(lattice).npz", cidag_cjs_exact = cidag_cjs_exact, cidag_cjs_bp = cidag_cjs,
    #    times= times,gate_app_times = gate_app_times, bp_times = bp_times)

    # Wide per-step timing CSV, matching the columns of the collaborator's run_timings.jl.
    mkpath(outdir)
    outfile = joinpath(outdir, "$(prefix)_chi$(χ).csv")
    write_wide_csv(outfile, rows)
    loop = filter(r -> r.step >= 1, rows)
    steptot = sort([r.single1 + r.hop + r.single2 + r.bp for r in loop])
    medtot = steptot[cld(length(steptot), 2)]
    medbp = sort([r.bp for r in loop])[cld(length(loop), 2)]
    medbytes = sort([r.single1_bytes + r.hop_bytes + r.single2_bytes + r.bp_bytes for r in loop])[cld(length(loop), 2)]
    println("χ=$(χ)  median step $(round(medtot; sigdigits=4))s  (bp $(round(medbp; sigdigits=4))s)  " *
            "heap $(round(medbytes / 2^20; sigdigits=4)) MiB/step  maxdim=$(maximum(r.maxdim for r in loop))  → $(basename(outfile))")
    return outfile
end

χs = [32]
mus =[1.0]
lattices = ["Hexagonal"]
for lattice in lattices
    for χ in χs
        for mu in mus
            main_fermions(χ, mu, lattice)
        end
    end
end

