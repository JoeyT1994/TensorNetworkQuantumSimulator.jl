using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using Statistics
using Dictionaries

using ITensors: Index, ITensor, inner, itensor, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str, Algorithm, datatype
using NamedGraphs.PartitionedGraphs: PartitionEdge

using Random

using EinExprs: Greedy
using Base.Threads
using LinearAlgebra

using Serialization
using Adapt: adapt

BLAS.set_num_threads(min(2, Sys.CPU_THREADS))
println("Julia is using " * string(nthreads()))
println("BLAS is using " * string(BLAS.get_num_threads()))
@show BLAS.get_config()

#Some gate definitions needed
function ITensors.op(
        ::OpName"S", ::SiteType"S=1/2"
    )
    mat = zeros(ComplexF64, 2, 2)
    mat[1, 1] = 1.0
    mat[2, 2] = 1.0 * im
    return mat
end

function ITensors.op(
        ::OpName"Sx", ::SiteType"S=1/2"
    )
    mat = zeros(ComplexF64, 2, 2)
    mat[1, 1] = 0.5 * (1.0 + 1.0 * im)
    mat[2, 2] = 0.5 * (1.0 + 1.0 * im)
    mat[1, 2] = 0.5 * (1.0 - 1.0 * im)
    mat[2, 1] = 0.5 * (1.0 - 1.0 * im)
    return mat
end

function ITensors.op(
        ::OpName"Sdg", ::SiteType"S=1/2"
    )
    mat = zeros(ComplexF64, 2, 2)
    mat[1, 1] = 1.0
    mat[2, 2] = -1.0 * im
    return mat
end

function ITensors.op(
        ::OpName"Sxdg", ::SiteType"S=1/2"
    )
    mat = zeros(ComplexF64, 2, 2)
    mat[1, 1] = 0.5 * (1.0 - 1.0 * im)
    mat[2, 2] = 0.5 * (1.0 - 1.0 * im)
    mat[1, 2] = 0.5 * (1.0 + 1.0 * im)
    mat[2, 1] = 0.5 * (1.0 + 1.0 * im)
    return mat
end

#Parse a QASM gate into a readable format for this repo (gate_str, qubits_acted_on, possible_kwarg)
function parse_gate(s::String)
    qs = [parse(Int64, q.match[2:(length(q.match) - 1)]) for q in eachmatch(r"\[(.*?)\]", s)]
    kwarg = match(r"\((.*?)\)", s)
    if !isnothing(kwarg)
        kwarg = eval(Meta.parse(kwarg.match[2:(length(kwarg.match) - 1)]))
    end

    gate_str = isnothing(kwarg) ? match(r"^[^\ ]+", s).match : match(r"^[^\(]+", s).match
    gate_str = uppercasefirst(gate_str)

    isnothing(kwarg) && return (gate_str, qs)
    return (gate_str, qs, kwarg)
end

#Read a full circuit from the given text file
function read_qasm_circuit(f::String)
    file = open(f)
    lines = readlines(file)
    close(file)
    n_lines = length(lines)
    gates = []
    for i in 4:n_lines
        line = lines[i]
        if !contains(line, "barrier")
            push!(gates, parse_gate(line))
        end
    end
    return gates
end

#Given a circuit of two-site and 1-site gates, build the graph induced by the circuit
function graph_from_circuit(circ)
    g = NamedGraph()
    for gate in circ
        qubits = gate[2]

        if length(qubits) == 2
            v1, v2 = first(qubits), last(qubits)
            !has_vertex(g, v1) && add_vertex!(g, v1)
            !has_vertex(g, v2) && add_vertex!(g, v2)
            !has_edge(g, NamedEdge(v1 => v2)) && add_edge!(g, NamedEdge(v1 => v2))
        end
    end
    return g
end

function split_circuit(circuit)
    U1, U2 = [], []
    split = false
    for gate in circuit
        if gate[1] == "Sx" || gate[1] == "S"
            split = true
        end

        if !split
            push!(U1, gate)
        else
            push!(U2, gate)
        end
    end

    return U1, U2
end

function measure(ψ::TensorNetworkState, z_vertices)
    ψO = TensorNetworkQuantumSimulator.QuadraticForm(ψ, v -> v ∈ z_vertices ? "Z" : "I")
    ψO = TN.update(BeliefPropagationCache(ψO))
    return TN.partitionfunction(ψO)
end

function monte_carlo_estimator(seed::Int, maxdim::Int, L::Int64, b::Float64, delta::Float64)
    ITensors.disable_warn_order()

    #Input file location, change to desired circuit
    root = "/Users/jtindall/Files/Data/Circuits/49q_circuits/"
    f = root * "49q_FL=" * string(L) * "_b=" * string(b) * "_delta=" * string(delta) * ".txt"


    circuit = read_qasm_circuit(f)

    U1, U2 = split_circuit(circuit)

    @show length(U1), length(U2)
    g = graph_from_circuit(circuit)

    #Look at Gates involved
    @show unique(first.(circuit))
    #Number of gates
    @show length(circuit)

    #Get the physical indices
    s = TN.siteinds("S=1/2", g)
    #Sites to initial "Z" on
    z_vertices = [52, 59, 72]


    #Random initial bitstring, use a seed
    Random.seed!(seed)

    #Apply kwargs (maximum bond dimension)
    apply_kwargs = (; maxdim = maxdim, cutoff = 1.0e-10, normalize_tensors = true)

    for i in 1:5
        println("Monte Carlo sample $i")
        initial_left_bitstring = Dictionary(vertices(g), [rand([-1, 1]) for v in vertices(g)])
        initial_right_bitstring = Dictionary(vertices(g), [rand([-1, 1]) for v in vertices(g)])

        ψ1 = tensornetworkstate(ComplexF32, v -> initial_left_bitstring[v] == -1 ? "Z-" : "Z+", g, s)
        χ1 = deepcopy(ψ1)

        ψ2 = tensornetworkstate(ComplexF32, v -> initial_right_bitstring[v] == -1 ? "Z-" : "Z+", g, s)
        χ2 = deepcopy(ψ2)

        ψ1, errs = apply_gates(U1, ψ1; apply_kwargs, verbose = false)
        ψ1 = TN.normalize(ψ1; alg = "bp")
        println("Average gate error was $(Statistics.mean(errs))")
        println("Rough fidility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

        ψ2, errs = apply_gates(U1, ψ2; apply_kwargs, verbose = false)
        ψ2 = TN.normalize(ψ2; alg = "bp")
        println("Average gate error was $(Statistics.mean(errs))")
        println("Rough fidility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

        χ1, errs = apply_gates(U2, χ1; apply_kwargs, verbose = false)
        χ1 = TN.normalize(χ1; alg = "bp")
        println("Average gate error was $(Statistics.mean(errs))")
        println("Rough fidility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

        χ2, errs = apply_gates(U2, χ2; apply_kwargs, verbose = false)
        χ2 = TN.normalize(χ2; alg = "bp")
        println("Average gate error was $(Statistics.mean(errs))")
        println("Rough fidility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

        for v in z_vertices
            TN.setindex_preserve!(ψ1, noprime(ψ1[v] * ITensors.op("Z", only(s[v]))), v)
            TN.setindex_preserve!(χ1, noprime(χ1[v] * ITensors.op("Z", only(s[v]))), v)
        end

        overlap1 = inner(ψ1, ψ2; alg = "bp")
        overlap2 = inner(χ1, χ2; alg = "bp")

        @show overlap1 * overlap2 * (4.0^49)
    end
    return
end


function main(seed::Int, maxdim::Int, L::Int64, b::Float64, delta::Float64)

    ITensors.disable_warn_order()

    #Input file location, change to desired circuit
    root = "/Users/jtindall/Files/Data/Circuits/49q_circuits/"
    f = root * "49q_FL=" * string(L) * "_b=" * string(b) * "_delta=" * string(delta) * ".txt"


    circuit = read_qasm_circuit(f)
    g = graph_from_circuit(circuit)

    #Look at Gates involved
    @show unique(first.(circuit))
    #Number of gates
    @show length(circuit)

    #Get the physical indices
    s = TN.siteinds("S=1/2", g)
    #Sites to initial "Z" on
    z_vertices = [52, 59, 72]


    #Random initial bitstring, use a seed
    Random.seed!(seed)
    initial_bitstring = Dictionary(vertices(g), [rand([-1, 1]) for v in vertices(g)])

    #Initial state based off bitstring
    ψ = tensornetworkstate(ComplexF32, v -> initial_bitstring[v] == -1 ? "Z-" : "Z+", g, s)

    ψ_bpc = BeliefPropagationCache(ψ)

    #Measure things
    TN.rescale!(ψ_bpc)
    O_init = measure(network(ψ_bpc), z_vertices)

    println("Initial value of O is $O_init")

    #Apply kwargs (maximum bond dimension)
    apply_kwargs = (; maxdim = maxdim, cutoff = 1.0e-10, normalize_tensors = true)

    #Apply the circuit to get O. U
    t = time()
    ψ_bpc, errs = apply_gates(circuit, ψ_bpc; apply_kwargs, verbose = false)
    t = time() - t
    println("Simulation O -> OU took $(t) seconds")
    println("Average gate error was $(Statistics.mean(errs))")
    println("Rough fidility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

    TN.rescale!(ψ_bpc)
    ψ = network(ψ_bpc)

    O_final = measure(ψ, z_vertices)
    @show O_final * O_init
    return O_final * O_init

end

signals = []
delta = 0.05
for i in 1:2
    signal = main(i, 128, 3, 0.25, delta)
    push!(signals, signal)
end


lambda = -log(Statistics.mean(signals)) / (delta * delta)

@show lambda
