using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

using TensorNetworkQuantumSimulator: AbstractTensorNetwork
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

function split_circuit(circuit, delta)
    U, V, Udag = [], [], []
    split = false
    for gate in circuit

        if gate[1] == "Rz" && gate[3] == 2 * delta
            push!(V, gate)
            split = true
        else
            if !split
                push!(U, gate)
            else
                push!(Udag, gate)
            end
        end
    end

    return U, V, Udag
end

function measure(ψ::TensorNetworkState, z_vertices)
    ψO = TensorNetworkQuantumSimulator.QuadraticForm(ψ, v -> v ∈ z_vertices ? "Z" : "I")
    ψO = TN.update(BeliefPropagationCache(ψO))
    return TN.partitionfunction(ψO)
end

function insert_linkinds!(tn::AbstractTensorNetwork)
    for e in edges(tn)
        if isempty(ITensors.commoninds(tn[src(e)], tn[dst(e)]))
            l = Index(1)
            TN.setindex_preserve!(tn, tn[src(e)] * onehot(l => 1), src(e))
            TN.setindex_preserve!(tn, tn[dst(e)] * onehot(l => 1), dst(e))
        end
    end
    return tn
end

function add_itensors(A::ITensor, B::ITensor)
    A_uniqueinds = ITensors.uniqueinds(A, B)
    B_uniqueinds = ITensors.uniqueinds(B, A)

    AplusB = ITensors.directsum(A => A_uniqueinds, B => B_uniqueinds)

    return first(AplusB)
end

function sim_linkinds!(tn::AbstractTensorNetwork)
    for e in edges(tn)
        lind = ITensors.commonind(tn[src(e)], tn[dst(e)])
        lind_sim = sim(lind)
        TN.setindex_preserve!(tn, ITensors.replaceind(tn[src(e)], lind, lind_sim), src(e))
        TN.setindex_preserve!(tn, ITensors.replaceind(tn[dst(e)], lind, lind_sim), dst(e))
    end
    return tn
end

function combine_linkinds!(tn::AbstractTensorNetwork)
    for e in edges(tn)
        linds = ITensors.commoninds(tn[src(e)], tn[dst(e)])
        if length(linds) > 1
            C = ITensors.combiner(linds)
            TN.setindex_preserve!(tn, tn[src(e)] * C, src(e))
            TN.setindex_preserve!(tn, tn[dst(e)] * C, dst(e))
        end
    end
    return tn
end

"""Add two itensornetworks together by growing the bond dimension. The network structures need to be have the same vertex names, same site index on each vertex """
function add(tn1::AbstractTensorNetwork, tn2::AbstractTensorNetwork)
    @assert issetequal(vertices(tn1), vertices(tn2))

    es = edges(tn1)
    tn12 = copy(tn1)
    new_edge_indices = Dict(
        zip(
            es,
            [
                Index(
                        dim(only(TN.virtualinds(tn1, e))) + dim(only(TN.virtualinds(tn2, e))),
                    ) for e in es
            ],
        ),
    )

    #Create vertices of tn12 as direct sum of tn1[v] and tn2[v]. Work out the matching indices by matching edges. Make index tags those of tn1[v]
    for v in vertices(tn1)
        es_v = filter(x -> src(x) == v || dst(x) == v, es)

        tn1v_linkinds = Index[only(TN.virtualinds(tn1, e)) for e in es_v]
        tn2v_linkinds = Index[only(TN.virtualinds(tn2, e)) for e in es_v]
        tn12v_linkinds = Index[new_edge_indices[e] for e in es_v]

        TN.setindex_preserve!(
            tn12, ITensors.directsum(
                tn12v_linkinds,
                tn1[v] => Tuple(tn1v_linkinds),
                tn2[v] => Tuple(tn2v_linkinds)
            ), v
        )
    end

    return tn12
end

function main(seed::Int, maxdim::Int, L::Int64, b::Float64, delta::Float64, nqubits::Int)

    ITensors.disable_warn_order()

    #Input file location, change to desired circuit
    root = "/Users/jtindall/Files/Data/Circuits/Algorithmic/" * string(nqubits) * "q/"
    f = root * "qa_A_FL=" * string(L) * "_b=0.25_delta=0.05.txt"


    circuit = read_qasm_circuit(f)

    U, V, Udag = split_circuit(circuit, 0.05)

    @show length(U), length(V), length(Udag)

    g = graph_from_circuit(circuit)

    #Look at Gates involved
    @show unique(first.(circuit))
    #Number of gates
    @show length(circuit)

    #Get the physical indices
    sphysical = TN.siteinds("S=1/2", g)
    sancilla = TN.siteinds("S=1/2", g)
    #Sites to initial "Z" on
    z_vertices = nqubits == 49 ? [92, 99, 112] : [52, 59, 72]

    U_circuit, Udag_circuit = TN.toitensor(U, sphysical), TN.toitensor(Udag, sancilla)
    V_vertices = [only(gate[2]) for gate in V]

    @show length(U_circuit), length(V), length(Udag_circuit)

    s_combined = Dictionary{NamedGraphs.vertextype(g), Vector{<:Index}}(collect(vertices(g)), [Index[only(sphysical[v]), only(sancilla[v])] for v in vertices(g)])
    U = TN.random_tensornetworkstate(g, s_combined)

    for v in vertices(g)
        if v ∉ V_vertices
            TN.setindex_preserve!(U, (1 / sqrt(2)) * ITensors.delta(s_combined[v]), v)
        else
            gate = only(filter(g -> only(g[2]) == v, V))
            gate = (gate[1], gate[2], 2 * delta)
            t = TN.toitensor(gate, sphysical)
            t = ITensors.replaceind(t, only(sphysical[v])', only(sancilla[v]))
            TN.setindex_preserve!(U, (1 / sqrt(2)) * t, v)
        end
    end

    insert_linkinds!(U)

    U = adapt(Vector{ComplexF32})(U)

    #Apply kwargs (maximum bond dimension)
    apply_kwargs = (; maxdim = maxdim, cutoff = 1.0e-14, normalize_tensors = false)

    circuit = [TN.toitensor(gate1, sphysical) * TN.toitensor(gate2, sancilla) for (gate1, gate2) in zip(reverse(U_circuit), Udag_circuit)]

    x = TN.norm_sqr(U; alg = "bp")
    @show x

    circuit = [adapt(Vector{ComplexF32})(g) for g in circuit]
    t = time()

    U_bpc = TN.update(TN.BeliefPropagationCache(U))
    U_bpc, errs = TN.apply_gates(circuit, U_bpc; apply_kwargs)
    t = time() - t
    println("Simulation V -> Udag . V .U took $(t) seconds")
    println("Average gate error was $(Statistics.mean(errs))")
    println("Rough fidility approximation (square overlap with correct state) is $(prod(1.0 .- errs))")

    #We want Tr(O.U.O.Udag)

    #Lets be clever
    #Frobenius norm of U:
    #U = TN.normalize(U; alg = "bp")
    U_bpc = TN.update(U_bpc)
    @show TN.partitionfunction(U_bpc)
    #U_bpc = TN.rescale(U_bpc)
    x = TN.partitionfunction(U_bpc)

    @show x
    U = network(U_bpc)
    UO = copy(U)

    for v in vertices(UO)
        if v ∈ z_vertices
            g = adapt(Vector{ComplexF32})(ITensors.op("Z", only(sphysical[v])))
            TN.setindex_preserve!(UO, noprime(g * UO[v]), v)
        end
    end

    OU = copy(U)
    for v in vertices(OU)
        if v ∈ z_vertices
            g = adapt(Vector{ComplexF32})(ITensors.op("Z", only(sancilla[v])))
            TN.setindex_preserve!(OU, noprime(g * OU[v]), v)
        end
    end

    # OU = sim_linkinds!(OU)
    # UOminusOU = add(OU, UO)
    # combine_linkinds!(UOminusOU)

    # @show TN.maxvirtualdim(UO)
    # @show TN.maxvirtualdim(UOminusOU)
    # y = TN.norm_sqr(UOminusOU; alg = "bp")
    # @show y

    # signal = x - 0.5*y
    # @show signal
    # log_signal = log(signal)
    #Build UO - OU


    #UO = TN.normalize(UO; alg = "bp")
    #OU = TN.normalize(OU; alg = "bp")

    #UO, OU = TN.symmetric_gauge(UO), TN.symmetric_gauge(OU)
    #signal_exact = TN.inner(UO, OU; alg = "exact")

    ρ_bpc = TN.BeliefPropagationCache(TN.BilinearForm(OU, UO))
    ρ_bpc = TN.update(ρ_bpc; maxiter = 50)
    signal = TN.partitionfunction(ρ_bpc)
    #@show signal, signal_exact

    return signal

end


function run()
    deltas = [0.0, 0.025]
    nq = 70
    signals = []
    for delta in deltas
        signal = main(1, 64, 6, 0.25, delta, nq)
        push!(signals, signal)
    end

    @show real.(signals), deltas
    lambda = -log(last(signals)) / (last(deltas) * last(deltas))

    return println("Approximate lambda is $(lambda)")
end

run()
