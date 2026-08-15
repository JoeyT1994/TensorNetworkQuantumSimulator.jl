using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
#using CUDA
using Dictionaries
using Graphs
using ITensors
using ITensors:
    @OpName_str, @SiteType_str, ITensor, Index, delta, denseblocks, op, replaceind,Algorithm
using LinearAlgebra
using NamedGraphs
using NamedGraphs.GraphsExtensions: subgraph
using Random
using Statistics
BLAS.set_num_threads(min(1, Sys.CPU_THREADS))

# ====================================== MPI setup ======================================= #

using MPI
MPI.Init()

const comm = MPI.COMM_WORLD
const root = 0

@assert MPI.Comm_size(comm) == 12 "This code is designed to run on 12 MPI ranks."

local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, 0)
my_rank = MPI.Comm_rank(local_comm)
#CUDA.device!(my_rank % CUDA.ndevices())

const RANK_QUBIT_MAP = Dictionary{Int, Int}(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [89, 91, 93, 107, 109, 111, 113, 127, 129, 131, 133, 151]
)
const QUBIT_RANK_MAP = Dictionary{Int, Int}(
    [89, 91, 93, 107, 109, 111, 113, 127, 129, 131, 133, 151],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
)

include("partition_graph.jl")

# --- gate matrices (S, Sx and daggers) ---
function ITensors.op(::OpName"S", ::SiteType"S=1/2")
    m = zeros(ComplexF64, 2, 2)
    m[1, 1] = 1
    m[2, 2] = im
    return m
end
function ITensors.op(::OpName"Sdg", ::SiteType"S=1/2")
    m = zeros(ComplexF64, 2, 2)
    m[1, 1] = 1
    m[2, 2] = -im
    return m
end
function ITensors.op(::OpName"Sx", ::SiteType"S=1/2")
    m = zeros(ComplexF64, 2, 2)
    m[1, 1] = 0.5 * (1 + im)
    m[2, 2] = 0.5 * (1 + im)
    m[1, 2] = 0.5 * (1 - im)
    m[2, 1] = 0.5 * (1 - im)
    return m
end
function ITensors.op(::OpName"Sxdg", ::SiteType"S=1/2")
    m = zeros(ComplexF64, 2, 2)
    m[1, 1] = 0.5 * (1 - im)
    m[2, 2] = 0.5 * (1 - im)
    m[1, 2] = 0.5 * (1 + im)
    m[2, 1] = 0.5 * (1 + im)
    return m
end

# --- QASM parsing (from heisenberg.jl) ---
function parse_gate(s::String)
    qs =
        [parse(Int64, q.match[2:(length(q.match) - 1)]) for q in eachmatch(r"\[(.*?)\]", s)]
    kwarg = match(r"\((.*?)\)", s)
    !isnothing(kwarg) &&
        (kwarg = eval(Meta.parse(kwarg.match[2:(length(kwarg.match) - 1)])))
    gate_str = isnothing(kwarg) ? match(r"^[^\ ]+", s).match : match(r"^[^\(]+", s).match
    gate_str = uppercasefirst(gate_str)
    isnothing(kwarg) && return (gate_str, qs)
    return (gate_str, qs, kwarg)
end
function read_qasm_circuit(f::String)
    lines = readlines(f)
    gates = []
    for i in 4:length(lines)
        !contains(lines[i], "barrier") && push!(gates, parse_gate(lines[i]))
    end
    return gates
end
function graph_from_circuit(circ)
    g = NamedGraph{Int}()
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

# --- Z-dephasing channel on vertex v (superket): O -> c² O + s² Z O Z ---
function dephasing_gate(v, φ, s)
    c2 = cos(φ / 2)^2
    s2 = sin(φ / 2)^2
    Ik = op("I", s[v][1])
    Ib = op("I", s[v][2])
    Zk = op("Z", s[v][1])
    Zb = op("Z", s[v][2])
    return c2 * (Ik * Ib) + s2 * (Zk * Zb)
end

# conjugate gate g*  (bra-leg gate for  O -> g O g†):  negate angle / swap dagger
const DAG = Dict("S" => "Sdg", "Sdg" => "S", "Sx" => "Sxdg", "Sxdg" => "Sx")
conj_gate(gate) = length(gate) == 3 ? (gate[1], gate[2], -gate[3]) : (DAG[gate[1]], gate[2])

# Build the observable O = Z^⊗(z_vertices) ⊗ I as a normalized operator-superket.
function build_O(g, s_combined, sphysical, sancilla, z_vertices, super_graph, bond_inds)
    O = TN.random_tensornetworkstate(g, s_combined)
    for v in vertices(g)
        if v ∉ z_vertices
            TN.setindex_preserve!(
                O,
                (1 / sqrt(2)) * denseblocks(ITensors.delta(s_combined[v])),
                v
            )
        else
            t = first(TN.toitensor(("Z", [v]), g, sphysical))
            t = replaceind(t, only(sphysical[v])', only(sancilla[v]))
            TN.setindex_preserve!(O, (1 / sqrt(2)) * t, v)
        end
    end
    TN.insert_partition_virtualinds!(O, super_graph, bond_inds)
    return O
end

# Full circuit as doubled conjugation gates: ket op(g) on sphysical, bra op(g*) on sancilla,
# plus a dephasing channel on both qubits after every 2-qubit gate.  Forward order.
function conjugation_circuit(circuit, g, sphysical, sancilla, s_combined; φ = 0.0)
    its = ITensor[]
    verts = Vector{Int}[]
    for gate in circuit
        ket, gate_verts = TN.toitensor(gate, g, sphysical)
        bra = first(TN.toitensor(conj_gate(gate), g, sancilla))
        push!(its, ket * bra)
        push!(verts, gate_verts)
        if length(gate[2]) == 2 && φ != 0.0
            for v in gate[2]
                push!(its, dephasing_gate(v, φ, s_combined))
                push!(verts, [v])
            end
        end
    end
    return its, verts
end

# split the delta=0.3 circuit into first-half U, perturbation V (Rz(2δ)), second-half Udag
function split_circuit(circuit, delta)
    U, V, Udag = [], [], []
    split = false
    for gate in circuit
        if gate[1] == "Rz" && abs(gate[3] - 2 * delta) <= 1.0e-10
            push!(V, gate)
            split = true
        else
            split ? push!(Udag, gate) : push!(U, gate)
        end
    end
    return U, V, Udag
end

function run_circuit_mpi(maxdim::Int, L::Int, sc::Float64; delta = 0.3)
    rank = MPI.Comm_rank(comm)
    ranks = Vector{Cint}(undef, MPI.Comm_size(comm))
    MPI.Allgather!(Cint[rank], ranks, comm)

    for gn in ("S", "Sdg", "Sx", "Sxdg")
        register_gate!(gn)
    end

    ITensors.disable_warn_order()

    η = (1 - sc) * 0.125
    dir =  "/Users/jtindall/.julia/dev/TensorNetworkQuantumSimulator/examples/"
    circuit = read_qasm_circuit(dir * "qa_A_FL=$(L)_sc=$(sc)_b=0.125_delta=$(delta).txt")
    U, V, _ = split_circuit(circuit, delta)

    super_graph = graph_from_circuit(circuit)

    regions, shared_vertices = partition_heavy_hex(super_graph, ranks)

    if rank == root
        sphysical = TN.siteinds("S=1/2", super_graph)
        sancilla = TN.siteinds("S=1/2", super_graph)
    else
        sphysical = nothing
        sancilla = nothing
    end

    sphysical = MPI.bcast(sphysical, comm; root)
    sancilla = MPI.bcast(sancilla, comm; root)

    s_combined = Dictionary{NamedGraphs.vertextype(super_graph), Vector{<:Index}}(
        collect(vertices(super_graph)),
        [Index[only(sphysical[v]), only(sancilla[v])] for v in vertices(super_graph)]
    )

    z_vertices = [131, 132, 133, 134, 135, 138, 139, 151, 152, 153, 154, 155]

    apply_kwargs = (; maxdim = maxdim, cutoff = 1.0e-14, normalize_tensors = false)
    Uc, Uv = conjugation_circuit(
        U,
        super_graph,
        sphysical,
        sancilla,
        s_combined
    )
    Vc, Vv = conjugation_circuit(
        V,
        super_graph,
        sphysical,
        sancilla,
        s_combined
    )

    subvertices = regions[rank]
    g = subgraph(super_graph, subvertices)

    sphysical = getindices(sphysical, Indices(subvertices))
    sancilla = getindices(sancilla, Indices(subvertices))
    s_combined = getindices(s_combined, Indices(subvertices))

    #Uc, Vc = CUDA.cu.(Uc), CUDA.cu.(Vc)
    bond_inds = TN.shared_bond_inds(super_graph; comm)
    O = build_O(g, s_combined, sphysical, sancilla, z_vertices, super_graph, bond_inds)
    #O = CUDA.cu(O)

    rank == root && println("Everything built: Applying gates")
    flush(stdout)

    blocked_gates!(true)
    O, errs, den = TN.apply_gates_mpi(
        Uc,
        O,
        super_graph,
        shared_vertices;
        gate_vertices = Uv,
        apply_kwargs,
        bp_update_kwargs = (; message_update_alg = Algorithm("blocked"))
    )  # A O A†

    M, _ = TN.apply_gates_mpi(
        Vc,
        O,
        super_graph,
        shared_vertices;
        update_cache = false,
        gate_vertices = Vv,
        apply_kwargs,
        bp_update_kwargs = (; message_update_alg = Algorithm("blocked"))
    ) # V 𝒪 V†

    rank == root && println("Circuit Applied")
    flush(stdout)

    # maxiter is explicit: the default comes from the local region, which is a tree, so it
    # would be 1 and the tolerance unreachable.
    num = real(TN.inner_mpi(O, M, super_graph, shared_vertices; comm, bp_update_kwargs = (; maxiter = 25, tolerance = 1.0e-5, message_update_alg = Algorithm("blocked"))))
    S = num / den
    fid = round(prod(1.0.-errs);sigdigits=3)
    rank == root && println(
        ">>> QUAD η=$η χ=$maxdim : 
        num=$(round(num; digits = 4)) \
        den(‖𝒪‖²)=$(round(den; digits = 4)) \
        S=$(round(S; digits = 4)) \
        fid≈$(round(fid; sigdigits = 3))"
    )
    flush(stdout)
    return S
end


Ls = [6]
scs = [0.6]
chi = 32
for L in Ls
    println("L is $L")
    for sc in scs
        run_circuit_mpi(chi, L, sc)
    end
end