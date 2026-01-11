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
using LinearAlgebra
using NPZ

using Adapt
using Dictionaries

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))

#Gate : rho -> rho .X. With this defined, expect(sqrt_rho, X; alg) = Tr(sqrt_rho . X sqrt_rho) / Tr(sqrt_rho sqrt_rho) = Tr(rho . X) / Tr(rho)
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


function ITensors.op(
    ::OpName"Z", ::SiteType"Pauli"
  )
    mat = zeros(ComplexF64, 4, 4)
    mat[1, 4] = 1
    mat[2, 3] = -im
    mat[3, 2] = im
    mat[4, 1] = 1
    return mat
end

function main()

    n = 8
    g = named_grid((n,n))
    #Pauli inds run over identity, X, Y, Z
    s = siteinds("Pauli", g)
    ρ = identitytensornetworkstate(ComplexF64, g, s)
    ITensors.disable_warn_order()
    maxdim = 4
    use_gpu =false

    δβ = 0.01 
    hx = 0.1
    J = 1.0

    # #Do a custom 4-way edge coloring then Trotterise the Hamiltonian into commuting groups
    ec1 = reduce(vcat, [[NamedEdge((j, i) => (j+1, i)) for j in 1:2:(n-1)] for i in 1:n])
    ec2 = reduce(vcat, [[NamedEdge((j, i) => (j+1, i)) for j in 2:2:(n-1)] for i in 1:n])
    ec3 = reduce(vcat, [[NamedEdge((i,j) => (i, j+1)) for j in 1:2:(n-1)] for i in 1:n])
    ec4 = reduce(vcat, [[NamedEdge((i,j) => (i, j+1)) for j in 2:2:(n-1)] for i in 1:n])
    ec = [ec1, ec2, ec3, ec4]

    @assert length(reduce(vcat, ec)) == length(edges(g))
    nsteps = 50
    apply_kwargs = (; maxdim = 4, cutoff = 1e-12)
    Js = Dictionary(collect(edges(g)), [first(src(e)) == first(dst(e)) && isodd(first(src(e))) ? -1.0 : 1.0 for e in edges(g)])
    MPS_message_rank = 16
    
    β = 0
    xs, zzs, βs = Float64[], Float64[], Float64[]
    vc=  first(center(g))
    vcn = first(neighbors(g, vc))
    for i in 1:nsteps
        #Apply the singsite rotations half way
        for v in vertices(g)
            gate, _ = TN.toitensor(("Rx", [v], -0.5 * im * hx *δβ), g, s)
            gate = adapt(datatype(ρ), gate)
            TN.setindex_preserve!(ρ, normalize(ITensors.apply(gate, ρ[v])), v)
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
                gate, _ = TN.toitensor(("Rzz", [src(pair), dst(pair)], -im * Js[pair] * δβ), g, s)
                gate = adapt(datatype(ρ), gate)
                envs = TN.incoming_messages(ρρ, [src(pair), dst(pair)])
                envs = adapt(datatype(ρ)).(envs)
                ρv1, ρv2  = TN.full_update(gate, ρ, [src(pair), dst(pair)]; envs, print_fidelity_loss = true, apply_kwargs...)
                TN.setindex_preserve!(ρ, normalize(ρv1), src(pair))
                TN.setindex_preserve!(ρ, normalize(ρv2), dst(pair))
            end
        end


        for v in vertices(g)
            gate, _ = TN.toitensor(("Rx", [v], -0.5 * im * hx *δβ), g, s)
            gate = adapt(datatype(ρ), gate)
            TN.setindex_preserve!(ρ, normalize(ITensors.apply(gate, ρ[v])), v)
        end

        β += δβ

        if use_gpu
            expecs = TN.expect(CUDA.cu(ρ), [("ZZ", [vc, vcn]), ("X", [vc])]; alg = "boundarymps", mps_bond_dimension = MPS_message_rank)
        else
            expecs = TN.expect(ρ,[("ZZ", [vc, vcn]), ("X", [vc])]; alg = "boundarymps", mps_bond_dimension = MPS_message_rank)
        end

        zz, x = first(expecs), last(expecs)

        push!(xs, real(x))
        push!(zzs, real(zz))
        push!(βs, β)
        println("Inverse Temperature is $β")
        println("Bond dimension of PEPO $(TN.maxvirtualdim(ρ))")

        println("Expectation value at beta  = $(2*β) of ZZ is $(zz)")
        println("Expectation value at beta  = $(2*β) of X is $(x)")
    end

    file_name = "TransverseFieldVillainIsingn$(n)h$(hx)Chi$(maxdim)"
    npzwrite("C:\\Users\\Joey\\Documents\\Data\\GBP\\VillainModel\\"*file_name*".npz", xs = xs, zzs = zzs, betas = βs)
end

main()