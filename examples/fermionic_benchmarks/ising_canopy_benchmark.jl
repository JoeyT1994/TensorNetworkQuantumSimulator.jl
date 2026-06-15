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

function main_ising(χ, hx, lattice; nsteps = 25, bp_iters = 30,
        outdir = joinpath(@__DIR__, "data"), prefix = "tnqs_spins")

    g = lattice == "Hexagonal" ? canopy_hexagonal_lattice(4,6) : named_comb_tree((8,6))
    s = siteinds("S=1/2", g)
    mod = 4
    ψ = tensornetworkstate(ComplexF64, v -> sum(v) % mod == 0 ? "Z+" : "Z-", g, s)

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

    println("Real time Evo in a staggered magnetic field on a lattice with $(nv(g)) vertices")

    J = 1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = nothing)
    #(Exponent of all these gates is -im * theta  where theta is the provided coefficient)
    single_site_gates = [isodd(v[3]) ? ("Rx", v, -hx*dt) : ("Rx", v, hx*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("Rzz", [src(e), dst(e)], 2*J*dt) for e in es])
    end

    t_update =0
    t_bp = 0

    v = first(center(g))
    vn = first(neighbors(g, v))
    @assert v ∈ vertices(g) && vn ∈ vertices(g)
    obs1, obs2 = [(["Z", "Z"], [v, vn])],[(["Z", "Z"], [vn, v])]
    ZiZjs = ComplexF64[only(expect(ψ_bpc, obs1)) + only(expect(ψ_bpc, obs2))]
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

        ZiZj = only(expect(ψ_bpc, obs1)) + only(expect(ψ_bpc, obs2))
        push!(ZiZjs, ZiZj)
        push!(bp_times, rb.time)
        push!(gate_app_times, r1.time + rh.time + r2.time)

        if i % 1 == 0
            println("Time is $(i*dt)")
            #println("This Time steps BP update took $(rb.time) secs")
            #println("This Time steps Gate app took $(r1.time + rh.time + r2.time) secs")
            #println("Current BD is $(maxvirtualdim(ψ_bpc))")
            #println("BP Measured hopping is $cidag_cj")
            #println("Exact hopping is $(cidag_cjexact)")
        end

    end

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
hxs =[1.0]
lattices = ["Hexagonal"]
for lattice in lattices
    for χ in χs
        for hx in hxs
            main_ising(χ, hx, lattice)
        end
    end
end

