using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics
using Base.Threads
using ITensors

function get_columns(L)
    pairs = []
    for i=1:L, j=1:L
        push!(pairs, [[(i,j),(i,k)] for k=j+1:L]...)
    end
    pairs
end

function main(L, χ, bmps_ranks; nl::Int=20,θh = 0)
    ITensors.disable_warn_order()
    g = named_grid((L,L); periodic =false)
    nq = length(vertices(g))

    #Define the gate parameters
    J = pi / 4

    layer = []

    Rx_layer = [("Rx", [v], θh) for v in vertices(g)]
    ec = edge_color(g, 4)

    Rzz_layer = []
    for edge_group in ec
        append!(Rzz_layer, ("Rzz", pair, -2*J) for pair in edge_group)
    end

    layer = vcat(Rx_layer, Rzz_layer)

    pairs = get_columns(L)
    verts = reshape([(i,j) for i=1:L,j=1:L],L^2)
    
    # the initial state (all up, use Float 32 precision)
    ψ0 = tensornetworkstate(ComplexF32, v -> "↑", g, "S=1/2")

    # max bond dimension for the TN
    apply_kwargs = (maxdim = χ, cutoff = 1.0e-12, normalize_tensors = false)

    # create the BP cache representing the square of the tensor network
    ψ_bpc = BeliefPropagationCache(ψ0)

    # an array to keep track of expectations taken via two methods
    bpc_states = []
    bmps_expects_z = [zeros(ComplexF64,L,L,nl) for r=bmps_ranks]
    bmps_expects_zz = [zeros(ComplexF64,length(pairs),nl) for r=bmps_ranks]

    # evolve! (First step takes long due to compilation)
    for l in 1:nl
        println("Layer $l")

        t1 = @timed ψ_bpc, errors =
            apply_gates(layer, ψ_bpc; apply_kwargs, verbose = false)

    	push!(bpc_states, copy(ψ_bpc))
	
        #Boundary MPS expectation
        ψ = network(ψ_bpc)

	@threads for r_i=1:length(bmps_ranks)
	    bmps_expects_zz[r_i][:,l] = expect(ψ, [(["Z","Z"], pair) for pair=pairs]; alg="boundarymps", mps_bond_dimension=bmps_ranks[r_i])
	    z = expect(ψ, [("Z", [v]) for v=verts]; alg="boundarymps", mps_bond_dimension=bmps_ranks[r_i])
	    bmps_expects_z[r_i][:,:,l] = reshape(z, (L,L))
	end

        println("    Took time: $(t1.time) [s]. Max bond dimension: $(maxvirtualdim(ψ_bpc))")
        println("    Maximum Gate error for layer was $(maximum(errors))"); flush(stdout)

    end
    return bpc_states, bmps_expects_z, bmps_expects_zz
end
