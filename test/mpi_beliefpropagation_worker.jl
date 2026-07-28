# Runs under mpiexec. One positional arg: case name in ("path", "ring", "chain3").
# Deliberately avoids Test and Random, which are in [extras] and so unavailable
# under --project=<package>.
using Dictionaries: Dictionary
using Graphs: neighbors, src, dst
using ITensors: ITensor
using MPI
using NamedGraphs: NamedEdge, rem_vertex!
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)

const FAILURES = String[]
function check(cond::Bool, msg::AbstractString)
    cond || push!(FAILURES, "[rank $RANK] $msg")
    return cond
end

function path_case()
    g = named_grid((6, 1))
    parts = [[(i, 1) for i in 1:3], [(i, 1) for i in 3:6]]
    shared = Dictionary([(3, 1)], [(Int32(0), Int32(1))])
    return (; name = "6-site path, 2 ranks", g, parts, shared, maxiter = 12)
end

function ring_case()
    g = named_grid((6, 1); periodic = true)
    parts = [[(i, 1) for i in 1:4], [(4, 1), (5, 1), (6, 1), (1, 1)]]
    shared = Dictionary([(1, 1), (4, 1)], [(Int32(0), Int32(1)), (Int32(0), Int32(1))])
    return (; name = "6-site ring, 2 ranks", g, parts, shared, maxiter = 200)
end

function chain3_case()
    g = named_grid((7, 1))
    parts = [[(i, 1) for i in 1:3], [(i, 1) for i in 3:5], [(i, 1) for i in 5:7]]
    shared = Dictionary([(3, 1), (5, 1)], [(Int32(0), Int32(1)), (Int32(1), Int32(2))])
    return (; name = "7-site path, 3 ranks", g, parts, shared, maxiter = 14)
end

const CASES = Dict("path" => path_case, "ring" => ring_case, "chain3" => chain3_case)

# All ranks must share byte-identical Index objects, so the network is built once
# and broadcast; Random.seed! does not reproduce ITensor Index ids.
function global_network(g)
    tn = RANK == 0 ? random_tensornetwork(Float64, g; bond_dimension = 2) : nothing
    return MPI.bcast(tn, COMM; root = 0)
end

function local_partition(tn, my_vertices)
    local_tn = copy(tn)
    for v in setdiff(collect(vertices(tn)), my_vertices)
        rem_vertex!(local_tn, v)
    end
    return local_tn
end

directed_edges(tn) = reduce(vcat, [[e, reverse(e)] for e in edges(tn)])

# The MPI constructor indexes `messages[edge]` directly, so entries must exist.
# Seed clean deltas rather than running a local update, whose boundary messages
# carry the dangling virtual index of the removed remote neighbour.
function seeded_cache(local_tn)
    bpc = BeliefPropagationCache(local_tn)
    des = directed_edges(local_tn)
    TNQS.setmessages!(bpc, des, [TNQS.default_message(bpc, e) for e in des])
    return bpc
end

function run_case(case)
    g = case.g
    tn = global_network(g)
    my_vertices = case.parts[RANK + 1]
    local_tn = local_partition(tn, my_vertices)
    bpc = seeded_cache(local_tn)
    mpi_bpc = TNQS.BeliefPropagationCacheMPI(bpc, g, case.shared; comm = COMM)

    check(mpi_bpc isa TNQS.BeliefPropagationCacheMPI, "constructed cache type")
    check(
        Set(collect(vertices(network(mpi_bpc)))) == Set(my_vertices),
        "local network vertices == partition"
    )

    # Every shared vertex this rank holds contributes one send edge (from its
    # local neighbour) and one recv edge (from the remote neighbour).
    expected_send, expected_recv = Set{NamedEdge}(), Set{NamedEdge}()
    for (sv, ranks) in pairs(case.shared)
        RANK in ranks || continue
        for n in neighbors(g, sv)
            e = NamedEdge(n => sv)
            push!(n in my_vertices ? expected_send : expected_recv, e)
        end
    end
    check(Set(keys(mpi_bpc.edges_to_send)) == expected_send, "edges_to_send == $expected_send")
    check(Set(keys(mpi_bpc.edges_to_recv)) == expected_recv, "edges_to_recv == $expected_recv")

    # Received messages must be present and carry the boundary tensor's dangling index.
    for e in expected_recv
        m = messages(mpi_bpc)[e]
        check(m isa ITensor, "received message on $e is an ITensor")
    end

    mpi_bpc = update(mpi_bpc; maxiter = case.maxiter, tolerance = nothing)
    check(mpi_bpc isa TNQS.BeliefPropagationCacheMPI, "cache survives update")
    return mpi_bpc
end

const CASE_NAME = ARGS[1]
run_case(CASES[CASE_NAME]())

# Agree on the verdict so every rank exits the same way and none is left blocking.
const TOTAL = MPI.Allreduce(length(FAILURES), MPI.SUM, COMM)
for f in FAILURES
    println(stderr, f)
end
RANK == 0 && println("case $CASE_NAME: $TOTAL failure(s)")
MPI.Finalize()
exit(TOTAL == 0 ? 0 : 1)
