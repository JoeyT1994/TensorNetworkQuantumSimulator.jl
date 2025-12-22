using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph

using LinearAlgebra: norm

using EinExprs: Greedy

using Random
Random.seed!(1634)

include("expect-corrected.jl")

function main(nx,ny)
    χ = 3
    ITensors.disable_warn_order()
    gs = [
        (named_grid((nx, 1)), "line", 0,-1),
        (named_grid((nx, ny)), "square", 4,11),
        (named_hexagonal_lattice_graph(nx, ny), "hexagonal", 6,11),
    ]

    states = []
    for (g, g_str, smallest_loop_size, wmax) in gs
        println("*****************************************")
        println("Testing for $g_str lattice with $(NG.nv(g)) vertices")
	wmax = min(wmax, NG.nv(g))
        ψ = TN.random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = χ)

        ψ = normalize(ψ; alg = "bp")
	ψIψ = BeliefPropagationCache(ψ)
	ψIψ = update(ψIψ)

	# BP expectation value
	v = first(center(g))
	expect_bp = real(expect(ψIψ, ("Z", [v])))
        expect_exact_v = real(expect(ψ, ("Z", [v]); alg = "exact"))
	clusters, egs, ig = TN.enumerate_clusters(g, wmax; must_contain=[v], min_deg = 1, min_v = smallest_loop_size)
	cluster_wts, expects = cluster_correlation(ψIψ,clusters, egs, ig, ("Z", [v]))


	regs = Dict()
	cnums = Dict()

	cc_wts = [1; smallest_loop_size:wmax;]
	for w=cc_wts
            regs[w],_,cnums[w]=TN.build_region_family_correlation(g,v,v,w)
        end

	expects_cc = Dict()
	for w=cc_wts
    	    expects_cc[w] = [real(cc_correlation(ψIψ,regs[w], cnums[w], ("Z", [v]))), real(cc_one_point_geometric(ψIψ,regs[w], cnums[w], ("Z", [v])))]
        end

        println("Bp expectation value for Z on site $(v) is $expect_bp")
	println("Cluster expansion expectation values: $(cluster_wts), $(real.(expects))")
	println("Cluster cumulant expansions: $(cc_wts), $([expects_cc[w] for w=cc_wts])")
        println("Exact expectation value is $expect_exact_v")

	println("***********************************")
	u = neighbors(g, v)[1]
	obs = (["Z","Z"], [u,v])
	expect_exact_u = real(expect(ψ, ("Z", [u]); alg = "exact"))
	expect_exact = real(expect(ψ,obs; alg = "exact")) - expect_exact_u * expect_exact_v
	println("Calculating connected correlation function between $(v) and $(u)")

	clusters, egs, ig = TN.enumerate_clusters(g, max(1,min(wmax,2*smallest_loop_size)); must_contain=[u,v], min_deg = 1, min_v = 2)

	cluster_wts, expects = cluster_correlation(ψIψ,clusters, egs, ig, obs)

	regs = Dict()
	cnums = Dict()

	cc_wts = [2;3:wmax;]
	for w=cc_wts
            regs[w],_,cnums[w]=TN.build_region_family_correlation(g,u,v,w)
        end

	expects_cc = Dict()
	for w=cc_wts
    	    expects_cc[w] = [real(cc_correlation(ψIψ,regs[w], cnums[w], obs)),real(cc_two_point_geometric(ψIψ,regs[w], cnums[w], obs))]
        end
	println("Cluster expansion expectation values: $(cluster_wts), $(real.(expects))")
	println("Cluster cumulant expansions: $(cc_wts), $([expects_cc[w] for w=cc_wts])")
	
        println("Exact expectation value is $expect_exact")
	push!(states, ψ)
    end
    return states
end