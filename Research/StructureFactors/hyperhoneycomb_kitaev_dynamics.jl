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

using Random
using TOML

using JLD2

include("utils.jl")

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

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

function two_time_correlator(ψ_init, ψ, ψψ::BeliefPropagationCache, v; kwargs...)
    posdef_cache_update_kwargs = (; maxiter = 20, tol = 1e-10, message_update_kwargs = (; message_update_function = TN.default_posdef_message_update_function))
    nonposdef_cache_update_kwargs = (; maxiter = 20, tol = 1e-10, message_update_kwargs = (; message_update_function = ITensorNetworks.default_message_update))
    ϕ = copy(ψ_init)
    ϕv = noprime(ϕ[v]*ITensors.op("Z", only(ITensorNetworks.siteinds(ϕ, v))))
    ϕ[v] = ϕv

    ϕϕnorm = inner(ϕ, ϕ; cache_update_kwargs = posdef_cache_update_kwargs, kwargs...)
    @show ϕϕnorm
    ψψnorm = scalar(ψψ; kwargs...)
    @show ψψnorm
    ψϕ = inner(ψ, ϕ; cache_update_kwargs = nonposdef_cache_update_kwargs, kwargs...)
    return ψϕ / sqrt(ϕϕnorm * ψψnorm)
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
    ψ0 = d["wavefunction"]
    s = siteinds(ψ0)
    g = ITensorNetworks.underlying_graph(ψ0)
    ec = edge_color(g, 3)

    ψ0ψ0 = build_bp_cache(ψ0)
    ψ0, ψ0ψ0 = normalize(ψ0, ψ0ψ0; update_cache = false)

    xx_observables, yy_observables, zz_observables = honeycomb_kitaev_heisenberg_observables(J, K, ec)
    ψ = deepcopy(ψ0)
    ψψ = deepcopy(ψ0ψ0)
    xxs, yys, zzs = expect(ψψ, xx_observables), expect(ψψ, yy_observables), expect(ψψ, zz_observables)
    e = sum(xxs) + sum(yys) + sum(zzs)
    println("Initial energy density is $(e/L)")

    vkick = first(center(g))
    δt = 0.01
    layer = honeycomb_kitaev_heisenberg_realtime_layer(J, K, δt, ec)
    ψvkick = noprime(ψ[vkick]*ITensors.op("Z", only(s[vkick])))
    ψ[vkick] = ψvkick
    ITensorNetworks.setindex_preserve_graph!(ψψ, ψvkick, (vkick, "ket"))
    ITensorNetworks.setindex_preserve_graph!(ψψ, dag(prime(ψvkick)), (vkick, "bra"))

    apply_kwargs = (; normalize = false, maxdim = maxdim, cutoff = 1e-12)

    no_steps = 100
    time = 0
    vns = neighbors(g, vkick)
    vnxx= only(filter(vn -> NamedEdge(vn => vkick) ∈ ec[1] || NamedEdge(vkick => vn) ∈ ec[1], vns))
    vnyy= only(filter(vn -> NamedEdge(vn => vkick) ∈ ec[2] || NamedEdge(vkick => vn) ∈ ec[2], vns))
    vnzz= only(filter(vn -> NamedEdge(vn => vkick) ∈ ec[3] || NamedEdge(vkick => vn) ∈ ec[3], vns))
    vnonnn = first(filter(vnp -> vnp != vkick, neighbors(g, vnxx)))
    #v = first(filter(vp -> vp != vkick, neighbors(g, vnzz)))
    v = vnxx
    #@assert length(a_star(g, v, vkick)) > 1
    σzt_σz0 = two_time_correlator(ψ0, ψ, ψψ, v)
    println("Initial two time correlator is $(σzt_σz0)")
    σz0_σz0 = expect(ψ0, ("ZZ", (vkick, v)); alg = "bp", cache! = Ref(ψ0ψ0))
    println("Initial two point expect is $(σz0_σz0)")

    for i in 1:no_steps
        ψ, ψψ, errs = apply(layer, ψ, ψψ; apply_kwargs)
        if i == 1
            ψ, ψψ = normalize(ψ, ψψ; update_cache = false)
        end
        time += δt
        phase = exp(im * time * e)
        σzt_σz0_bp = two_time_correlator(ψ0, ψ, ψψ, v; alg = "bp")
        σzt_σz0_lc = two_time_correlator(ψ0, ψ, ψψ, v; alg = "loopcorrections", max_configuration_size = 12)
        println("Time is $time")
        println("Maximum bond dimension is $(maxlinkdim(ψ))")
        println("Maximum gate error is $(maximum(errs))")
        println("Two time correlator (bp computed) is $(phase * σzt_σz0_bp)")
        println("Two time correlator (lc computed) is $(phase * σzt_σz0_lc)")
    end



    return nothing
end

maxdim_gs, maxdim =10,10
i = 10
main(i, maxdim_gs, maxdim)
