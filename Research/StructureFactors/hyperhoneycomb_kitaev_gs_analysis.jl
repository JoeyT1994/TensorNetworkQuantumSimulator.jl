using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
using ITensorNetworks: IndsNetwork, BeliefPropagationCache
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: add_edges, add_vertices
using Graphs: center

using Random
using TOML

using JLD2

include("utils.jl")

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

using Dictionaries

#Construct a graph with edges everywhere a two-site gate appears.
function build_graph_from_interactions(list; sort_vertices = false)
    vertices = []
    edges = []
    for term in list
        vsrc, vdst = (term[3],), (term[4],)
        if vsrc ∉ vertices
            push!(vertices, vsrc)
        end
        if vdst ∉ vertices
            push!(vertices, vdst)
        end
        e = NamedEdge(vsrc => vdst)
        if e ∉ edges || reverse(e) ∉ edges
            push!(edges, e)
        end
    end
    g = NamedGraph()
    if sort_vertices
      vertices = sort(vertices; by = v -> first(v))
    end
    g = add_vertices(g, vertices)
    g = add_edges(g, edges)
    return g
end
  
function hyperhoneycomb_graph(L; kwargs...)
      file = pwd()*"/Research/StructureFactors/Data/hyperhoneycomb."*string(L)*".pbc.HB.Kitaev.nosyminfo.toml"
      data = TOML.parsefile(file)
      interactions = data["Interactions"]
      heisenberg_interactions = filter(d -> first(d) == "HB", interactions)
      g = build_graph_from_interactions(heisenberg_interactions; kwargs...)
      return g
end

function obs_optimised(ψψ, K::Float64, J::Float64, ec; kwargs...)
    x_edge, y_edge, z_edge = first(ec[1]), first(ec[2]), first(ec[3])
    zz_obs = [("ZZ", (src(x_edge), dst(x_edge))), ("ZZ", (src(y_edge), dst(y_edge))), ("ZZ", (src(z_edge), dst(z_edge)))]
    xx_obs = [("XX", (src(x_edge), dst(x_edge))), ("XX", (src(y_edge), dst(y_edge))), ("XX", (src(z_edge), dst(z_edge)))]
    cache_update_kwargs = (;  maxiter = 25, tol = 1e-14, message_update_kwargs = (; message_update_function = ms -> make_eigs_real.(ITensorNetworks.default_message_update(ms))))

    zz_expects = Float64[]
    for obs in zz_obs
        numer_ps, denom_ps = [], []
        for proj in [("Z+Z+", obs[2], 1.0), ("Z-Z-", obs[2], 1.0), ("Z+Z-", obs[2], -1.0), ("Z-Z+", obs[2], -1.0)]
            ψPψ = TN.insert_projector(ψψ, proj)
            ψPψ = updatecache(ψPψ; cache_update_kwargs...)
            p = scalar(ψPψ; kwargs...)
            push!(numer_ps, last(proj)*p)
            push!(denom_ps, p)
        end
        exp = sum(numer_ps) / sum(denom_ps)
        push!(zz_expects, exp)
    end

    xx_expects = Float64[]
    for obs in xx_obs
        numer_ps, denom_ps = [], []
        for proj in [("X+X+", obs[2], 1.0), ("X-X-", obs[2], 1.0), ("X+X-", obs[2], -1.0), ("X-X+", obs[2], -1.0)]
            ψPψ = TN.insert_projector(ψψ, proj)
            ψPψ = updatecache(ψPψ; cache_update_kwargs...)
            p = scalar(ψPψ; kwargs...)
            push!(numer_ps, last(proj)*p)
            push!(denom_ps, p)
        end
        exp = sum(numer_ps) / sum(denom_ps)
        push!(xx_expects, exp)
    end
    return zz_expects, xx_expects
end



function main(i::Int64, maxdim_gs::Int64, maxdim::Int64)

    #Get the graph and interactions from the .tomls. Flag ensures Vertices are ordered consistent with the .toml file
    L = 128

    n = 144
    θ = (2 * pi * i) / (n)
    K, J = 2*sin(θ), cos(θ)
    println("Beginning simulation with theta = $(θ), J = $(J), K = $(K) and a maxdim of $(maxdim).")

    file_name = "L"*string(L)*"i"*string(i)*"maxdim"*string(maxdim_gs)
    f = "/mnt/home/jtindall/ceph/Data/StructureFactors/Hyperhoneycomb/GroundStateWavefunctions/"*file_name*".jld2"
    d = load(f)
    ψ = d["wavefunction"]
    ψψ = build_bp_cache(ψ)
    ψ, ψψ = normalize(ψ, ψψ; update_cache = false)
    g = ITensorNetworks.underlying_graph(ψ)
    ec = edge_color(g, 3)

    xx_observables, yy_observables, zz_observables = honeycomb_kitaev_heisenberg_observables(J, K, ec)

    x_obs, y_obs, z_obs = [("X", [v]) for v in vertices(g)], [("Y", [v]) for v in vertices(g)], [("Z", [v]) for v in vertices(g)]

    xxs, yys, zzs = expect(ψψ, xx_observables), expect(ψψ, yy_observables), expect(ψψ, zz_observables)
    e_bp_density = (sum(xxs) + sum(yys) + sum(zzs))/L

    zzs_naive, xxs_naive = Float64[expect(ψψ, ("ZZ", first(ec[1]))), expect(ψψ, ("ZZ", first(ec[2]))), expect(ψψ, ("ZZ", first(ec[3])))],
     Float64[expect(ψψ, ("XX", first(ec[1]))), expect(ψψ, ("XX", first(ec[2]))), expect(ψψ, ("XX", first(ec[3])))]

    println("BP computed zzs (naive) are $(zzs_naive)")

    println("BP Energy is $e_bp_density")

    zzs_bp, xxs_bp = obs_optimised(ψψ, K, J, ec; alg = "bp")

    println("BP computed zzs are $(zzs_bp)")

    zzs_loopcorrected, xxs_loopcorrected = obs_optimised(ψψ, K, J, ec; alg = "loopcorrections", max_configuration_size = 12)

    println("Loop Corrected zzs are $(zzs_loopcorrected)")

    xs, ys, zs = expect(ψψ, x_obs), expect(ψψ, y_obs), expect(ψψ, z_obs)
    sx, sy, sz = sum(xs), sum(ys), sum(zs)

    M = sqrt(sx*sx + sy*sy + sz*sz) / L

    M_sq = sum([sqrt(x*x + y*y + z*z) for (x,y,z) in zip(xs,ys,zs)]) / L

    println("Final Spin is $M")

    println("Final Squared Spin is $M_sq")

    return M, M_sq, e_bp_density, zzs_naive, zzs_bp, zzs_loopcorrected, xxs_naive, xxs_bp, xxs_loopcorrected
end

i, maxdim_gs = parse(Int64, ARGS[1]), parse(Int64, ARGS[2])
#i, maxdim_gs = 21, 5
maxdim = maxdim_gs
M, M_sq, e_bp_density, zzs_naive, zzs_bp, zzs_loopcorrected, xxs_naive, xxs_bp, xxs_loopcorrected = main(i, maxdim_gs, maxdim)
npzwrite("/mnt/home/jtindall/ceph/Data/StructureFactors/Hyperhoneycomb/GroundStateProperties/i$(i)maxdim$(maxdim_gs).npz", M, M_sq, e_bp_density, zzs_naive, zzs_bp, zzs_loopcorrected, xxs_naive, xxs_bp, xxs_loopcorrected)