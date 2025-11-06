using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensors: @OpName_str, @SiteType_str, Algorithm, datatype, ITensors

using NamedGraphs: NamedGraphs, edges, NamedEdge
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: add_edges, add_vertices

using Random
using TOML

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

using CUDA

using Adapt
using Dictionaries
using JLD2

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))

DATA_DIR = "/mnt/ceph/users/gsommers/data/"

#Gate : rho -> rho .X
function ITensors.op(
    ::OpName"X", ::SiteType"Pauli"
  )
    mat = zeros(ComplexF64, 4,4)
    mat[1,2] = 1
    mat[2,1] = 1
    mat[3,4] = im
    mat[4,3] = -im
    return mat
end

function prep_edges(n::Int, g::AbstractNamedGraph)
    # #Do a custom 4-way edge coloring then Trotterize the Hamiltonian into commuting groups
    ec1 = reduce(vcat, [[NamedEdge((j, i) => (j+1, i)) for j in 1:2:(n-1)] for i in 1:n])
    ec2 = reduce(vcat, [[NamedEdge((j, i) => (j+1, i)) for j in 2:2:(n-1)] for i in 1:n])
    ec3 = reduce(vcat, [[NamedEdge((i,j) => (i, j+1)) for j in 1:2:(n-1)] for i in 1:n])
    ec4 = reduce(vcat, [[NamedEdge((i,j) => (i, j+1)) for j in 2:2:(n-1)] for i in 1:n])
    ec = [ec1, ec2, ec3, ec4]

    @assert length(reduce(vcat, ec)) == length(edges(g))
    ec
end

# apply layer of single qubit gates
function apply_single_qubit_layer!(ρ::TensorNetworkState, gates::Dict)
    for v=keys(gates)
        setindex_preserve_graph!(ρ, normalize(ITensors.apply(gates[v], ρ[v])), v)
    end
end

#Apply the two site rotations, use a boundary MPS cache to apply them (need to run column or row wise depending on the gates)
function apply_two_qubit_layer!(ρ::TensorNetworkState, ec::Array, gates::Dict; MPS_message_rank::Int, use_gpu::Bool=true, apply_kwargs...)
    for (k, colored_edges) in enumerate(ec)

        #Only if you want to use GPU to do boundary MPS
	println("Starting boundary MPS cache")
        if use_gpu
            @time ρ_gpu =CUDA.cu(ρ)
            @time ρρ = TN.BoundaryMPSCache(ρ_gpu, MPS_message_rank; partition_by = (k== 1 || k == 2) ? "col" : "row", gauge_state = false)
        else
            @time ρρ = TN.BoundaryMPSCache(ρ, MPS_message_rank; partition_by = (k== 1 || k == 2) ? "col" : "row", gauge_state = false)
        end
        @time ρρ = TN.update(ρρ)
        @time TN.update_partitions!(ρρ, collect(TN.partitionvertices(TN.supergraph(ρρ))))

	println("Starting two-qubit gates")
	@time begin
	    for pair in colored_edges
	        apply_two_qubit_gate!(ρ,ρρ, gates[pair], pair; apply_kwargs...)
            end
	end
    end
end
   
function apply_two_qubit_gate!(ρ::TensorNetworkState,ρρ::TN.BoundaryMPSCache, gate::ITensor, pair::NamedEdge; apply_kwargs...)
    envs = TN.incoming_messages(ρρ, [src(pair), dst(pair)])
    envs = adapt(datatype(ρ)).(envs)
    ρv1, ρv2  = TN.full_update(gate, ρ, [src(pair), dst(pair)]; envs, print_fidelity_loss = true, apply_kwargs...)
    TN.setindex_preserve!(ρ, normalize(ρv1), src(pair))
    TN.setindex_preserve!(ρ, normalize(ρv2), dst(pair))
end

function intermediate_save(sqrtρ, β; δβ::Float64, χ::Int, n::Int, MPS_message_rank, save_tag = "", hx = -3.04438)
    dat = Dict("L"=>n, "δβ"=>δβ, "β"=>β, "χ"=>χ, "sqrtρ"=>sqrtρ, "mps_rank"=>MPS_message_rank, "hx"=>hx)
    save(DATA_DIR * "$(save_tag)L$(n)_χ$(χ)_D$(MPS_message_rank)_step$(round(δβ,digits=3))_$(round(β,digits=3)).jld2", dat)
end

function intermediate_save_bp(ρ, errs, β; δβ::Float64, χ::Int, n::Int, save_tag = "", hx = -3.04438)
    dat = Dict("L"=>n, "δβ"=>δβ, "β"=>β, "χ"=>χ, "ρ"=>ρ, "errs"=>errs, "hx"=>hx)
    dat["X"] = 	expect(ρ, [("X", [v]) for v=vertices(network(ρ).tensornetwork.data_graph.underlying_graph)])
    save(DATA_DIR * "$(save_tag)L$(n)_χ$(χ)_step$(round(δβ,digits=3))_$(round(β,digits=3)).jld2", dat)
end

function expect_bmps(dat::Dict; obs = "X", MPS_message_rank::Int = 10, save_tag = "", use_gpu::Bool = true, start_i::Int = 1)
    all_verts = collect(vertices(dat["sqrtρ"][1].tensornetwork.data_graph.underlying_graph))
    expect_vals = zeros(length(all_verts), length(dat["sqrtρ"])-start_i+1)
    for i=start_i:length(dat["sqrtρ"])
        if use_gpu
            sqrtρ = CUDA.cu(dat["sqrtρ"][i])
	else
	    sqrtρ = dat["sqrtρ"][i]
	end
	@time expect_vals[:,i-start_i+1] = real.(TN.expect(sqrtρ, [(obs, [v]) for v=all_verts]; alg = "boundarymps", mps_bond_dimension = MPS_message_rank))
	 save(DATA_DIR * "$(save_tag)L$(dat["L"])_χ$(dat["χ"])_D$(MPS_message_rank)_step$(dat["δβ"])_$(dat["β"][i]).jld2", Dict(obs=>expect_vals[:,i-start_i+1], "verts"=>all_verts, "hx"=>dat["hx"], "β"=>dat["β"][i], "χ"=>[dat["χ"],maxlinkdim(dat["sqrtρ"][i])], "mps_rank"=>MPS_message_rank, "δβ"=>dat["δβ"], "L"=>dat["L"]))
	 flush(stdout)
    end
    all_verts, expect_vals
end

function evolve_bmps(n::Int, nsteps::Int; hx=-3.04438, δβ = 0.01, use_gpu::Bool = true, χ::Int=4, MPS_message_rank::Int = 10, save_tag = "")

    g = named_grid((n,n))
    s = siteinds("Pauli", g)
    ρ = identitytensornetworkstate(ComplexF64, g, s)
    evolve_bmps(ρ, n, nsteps; β=0, hx=hx, δβ=δβ, use_gpu = use_gpu, χ=χ, MPS_message_rank = MPS_message_rank, save_tag = save_tag)
    
end

function evolve_bmps(ρ::TensorNetworkState, n::Int, nsteps::Int; β = 0, hx=-3.04438, δβ = 0.01, use_gpu::Bool=true, χ::Int=4, MPS_message_rank::Int = 10, save_tag = "")
    g = ρ.tensornetwork.data_graph.underlying_graph
    s = siteinds(ρ)
    ITensors.disable_warn_order()

    J = -1

    ec = prep_edges(n, g)
    apply_kwargs = (; maxdim = χ, cutoff = 1e-12)

    two_qubit_gates = Dict(pair=>adapt(datatype(ρ), TN.toitensor(("Rzz", [src(pair), dst(pair)], -0.5*im * J * δβ), s)) for pair=vcat(ec...))

    single_qubit_gates = Dict(v=>adapt(datatype(ρ), TN.toitensor(("Rx", [v], -0.25 * im * hx *δβ), s)) for v=vertices(g))
    
    for i in 1:nsteps
        #Apply the singsite rotations half way
	apply_single_qubit_layer!(ρ, single_qubit_gates)

        @time apply_two_qubit_layer!(ρ, ec, two_qubit_gates; MPS_message_rank = MPS_message_rank, use_gpu = use_gpu, apply_kwargs...)

        apply_single_qubit_layer!(ρ, single_qubit_gates)

        β += δβ

        println("Inverse Temperature is $(β)"); flush(stdout)

	intermediate_save(ρ,β; χ=χ,n=n,MPS_message_rank = MPS_message_rank, δβ=δβ, hx=hx, save_tag = save_tag)
    end
end

function evolve_bp(n::Int, nsteps::Int; hx=-3.04438, δβ = 0.01, use_gpu::Bool=true, χ::Int=4, save_tag = "")
    g = named_grid((n,n))
    s = siteinds("Pauli", g)
    ρ = identitytensornetworkstate(ComplexF64, g, s)
    ρρ = BeliefPropagationCache(ρ)
    evolve_bp(ρρ, n, nsteps; β=0, hx=hx, δβ=δβ, use_gpu = use_gpu, χ=χ, save_tag = save_tag)
end

function evolve_bp(ρρ::BeliefPropagationCache, n::Int, nsteps::Int; β = 0, hx=-3.04438, δβ = 0.01, use_gpu::Bool=true, χ::Int=4, save_tag = "")
    g = network(ρρ).tensornetwork.data_graph.underlying_graph
    s = siteinds(network(ρρ))
    ITensors.disable_warn_order()

    J = -1

    ec = prep_edges(n, g)
    apply_kwargs = (; maxdim = χ, cutoff = 1e-12)
    
    two_qubit_gates = [adapt(datatype(network(ρρ)), TN.toitensor(("Rzz", [src(pair), dst(pair)], -0.5*im * J * δβ), s)) for pair=vcat(ec...)]

    single_qubit_gates = [adapt(datatype(network(ρρ)), TN.toitensor(("Rx", [v], -0.25 * im * hx *δβ), s)) for v=vertices(g)]

    layer = vcat(single_qubit_gates, two_qubit_gates, single_qubit_gates)
    for i in 1:nsteps

    	@time ρρ, errs = apply_gates(layer, ρρ; apply_kwargs, verbose = false)

        β += δβ

        println("Inverse Temperature is $(β)"); flush(stdout)
	intermediate_save_bp(ρρ,errs,β; χ=χ,n=n, δβ=δβ, hx=hx, save_tag = save_tag)
    end
end
