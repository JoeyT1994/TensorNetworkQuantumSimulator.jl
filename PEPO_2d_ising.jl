using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors
using ITensors: @OpName_str, @SiteType_str, Algorithm, datatype

using ITensorNetworks: AbstractBeliefPropagationCache, IndsNetwork, setindex_preserve_graph!
using NamedGraphs
using NamedGraphs: edges
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

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))

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

function main()

    n = 6
    g = named_grid((n,n))
    s = siteinds(g, "Pauli")
    ρ = identitytensornetworkstate(ComplexF64, g, s)
    ITensors.disable_warn_order()

    use_gpu = true

    δβ = 0.01 
    hx = -3.1
    J = -1

    # #Do a custom 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    ec1 = reduce(vcat, [[NamedEdge((j, i) => (j+1, i)) for j in 1:2:(n-1)] for i in 1:n])
    ec2 = reduce(vcat, [[NamedEdge((j, i) => (j+1, i)) for j in 2:2:(n-1)] for i in 1:n])
    ec3 = reduce(vcat, [[NamedEdge((i,j) => (i, j+1)) for j in 1:2:(n-1)] for i in 1:n])
    ec4 = reduce(vcat, [[NamedEdge((i,j) => (i, j+1)) for j in 2:2:(n-1)] for i in 1:n])
    ec = [ec1, ec2, ec3, ec4]

    @assert length(reduce(vcat, ec)) == length(edges(g))
    nsteps = 50
    apply_kwargs = (; maxdim = 4, cutoff = 1e-12)

    MPS_message_rank = 10
    
    β = 0
    for i in 1:nsteps
        #Apply the singsite rotations half way
        for v in vertices(g)
            gate = TN.toitensor(("Rx", [v], -0.25 * im * hx *δβ), s)
            gate = adapt(datatype(ρ), gate)
            setindex_preserve_graph!(ρ, normalize(apply(gate, ρ[v])), v)
        end

        #Apply the two site rotations, use a boundary MPS cache to apply them (need to run column or row wise depending on the gates)
        for (k, colored_edges) in enumerate(ec)

            #Only if you want to use GPU to do boundary MPS
            if use_gpu
                ρ_gpu =CUDA.cu(ρ)
                ρρ = TN.BoundaryMPSCache(ρ_gpu, MPS_message_rank; partition_by = (k== 1 || k == 2) ? "col" : "row", gauge_state = false)
            else
                ρρ = TN.BoundaryMPSCache(ρ, MPS_message_rank; partition_by = (k== 1 || k == 2) ? "col" : "row", gauge_state = false)
            end
            ρρ = TN.update(ρρ)
            TN.update_partitions!(ρρ, collect(TN.partitionvertices(TN.supergraph(ρρ))))

            for pair in colored_edges
                gate = TN.toitensor(("Rzz", [src(pair), dst(pair)], -0.5*im * J * δβ), s)
                gate = adapt(datatype(ρ), gate)
                envs = TN.incoming_messages(ρρ, [src(pair), dst(pair)])
                envs = adapt(datatype(ρ)).(envs)
                ρv1, ρv2  = ITensorNetworks.full_update_bp(gate, TN.tensornetwork(ρ), [src(pair), dst(pair)]; envs, apply_kwargs...)
                ρ[src(pair)], ρ[dst(pair)] = normalize(ρv1), normalize(ρv2)
            end
        end


        for v in vertices(g)
            gate = TN.toitensor(("Rx", [v], -0.25 * im * hx *δβ), s)
            gate = adapt(datatype(ρ), gate)
            setindex_preserve_graph!(ρ, normalize(apply(gate, ρ[v])), v)
        end

        β += δβ

        if use_gpu
            sz_doubled = TN.expect(ρ, ("X", [(2,2)]); alg = "boundarymps", mps_bond_dimension = MPS_message_rank)
        else
            sz_doubled = TN.expect(CUDA.cu(ρ), ("X", [(2,2)]); alg = "boundarymps", mps_bond_dimension = MPS_message_rank)
        end

        println("Inverse Temperature is $β")
        println("Bond dimension of PEPO $(ITensorNetworks.maxlinkdim(ρ))")

        println("Expectation value at beta  = $(2*β) is $(sz_doubled)")
    end


end

main()