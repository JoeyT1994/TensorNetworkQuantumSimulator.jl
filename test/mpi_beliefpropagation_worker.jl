# Runs under mpiexec. One positional arg: a case name from CASES below.
# Deliberately avoids Test and Random, which are in [extras] and so unavailable
# under --project=<package>.
using Dictionaries: Dictionary, delete!
using Graphs: dst, neighbors, src
using ITensors: ITensors, ITensor, contract, inds, plev, scalar, state
using LinearAlgebra: norm
using MPI
using NamedGraphs.GraphsExtensions: subgraph
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

# Partitions are DISJOINT: every vertex is owned by exactly one rank. `ranks` is the owner map the
# cache constructor takes.
function partitioned(g, parts; kwargs...)
    ranks = Dictionary(
        reduce(vcat, parts),
        reduce(vcat, [fill(Int32(r - 1), length(p)) for (r, p) in enumerate(parts)])
    )
    return (; g, parts, ranks, kwargs...)
end

function path_case()
    g = named_grid((6, 1))
    parts = [[(i, 1) for i in 1:3], [(i, 1) for i in 4:6]]
    return partitioned(g, parts; name = "6-site path, 2 ranks", maxiter = 12)
end

# The one loopy case, so the only one whose BP convergence is not automatic -- see `fixed_entries!`,
# without which roughly one draw in thirty sits in a period-2 cycle that no `maxiter` breaks.
function ring_case()
    g = named_grid((6, 1); periodic = true)
    parts = [[(i, 1) for i in 1:3], [(i, 1) for i in 4:6]]
    return partitioned(g, parts; name = "6-site ring, 2 ranks", maxiter = 200)
end

function chain3_case()
    g = named_grid((7, 1))
    parts = [[(i, 1) for i in 1:3], [(i, 1) for i in 4:5], [(i, 1) for i in 6:7]]
    return partitioned(g, parts; name = "7-site path, 3 ranks", maxiter = 14)
end

# Cut across the spine, where the boundary vertices have degree 3. Every 1-D case above is wide, so
# without this the thin-QR branch of `absorb_boundary_in!` is never exercised.
function comb_case()
    g = named_comb_tree((4, 3))
    spine(i) = [(i, j) for j in 1:3]
    parts = [reduce(vcat, [spine(i) for i in 1:2]), reduce(vcat, [spine(i) for i in 3:4])]
    return partitioned(g, parts; name = "4x3 comb tree, 2 ranks", maxiter = 20)
end

apply_path_case() = (; path_case()..., name = "6-site path apply, 2 ranks", maxiter = 25)
apply_ring_case() = (; ring_case()..., name = "6-site ring apply, 2 ranks", maxiter = 100)
apply_comb_case() = (; comb_case()..., name = "4x3 comb tree apply, 2 ranks", maxiter = 25)

# Fixed entries in place of the drawn ones, keeping only the index structure. Positive, so every
# transfer matrix is positive and the BP fixed point the comparisons below assume is unique.
function fixed_entries!(tn)
    for (i, v) in enumerate(sort(collect(vertices(tn)); by = repr))
        d = ITensors.data(tn[v])
        for k in eachindex(d)
            r = 1 + 0.3 * cospi(0.37 * (k + 7i))
            d[k] = eltype(d) <: Complex ? r * cispi(0.11 * cospi(0.23 * (k + 5i))) : r
        end
    end
    return tn
end

fixed_entries!(ψ::TensorNetworkState) = (fixed_entries!(TNQS.tensornetwork(ψ)); ψ)

# All ranks must share byte-identical Index objects, so the network is built once
# and broadcast; Random.seed! does not reproduce ITensor Index ids.
function global_network(g)
    tn = RANK == 0 ?
        fixed_entries!(random_tensornetwork(Float64, g; bond_dimension = 2)) : nothing
    return MPI.bcast(tn, COMM; root = 0)
end

function global_state(g)
    ψ = RANK == 0 ?
        fixed_entries!(random_tensornetworkstate(ComplexF64, g; bond_dimension = 2)) : nothing
    return MPI.bcast(ψ, COMM; root = 0)
end

# Index ids are random, so they must be drawn once and broadcast.
function global_siteinds(g)
    s = RANK == 0 ? siteinds("S=1/2", g) : nothing
    return MPI.bcast(s, COMM; root = 0)
end

function restrict(d::Dictionary, ks)
    out = copy(d)
    for k in setdiff(collect(keys(out)), ks)
        delete!(out, k)
    end
    return out
end

function local_partition(tn, my_vertices)
    local_tn = copy(tn)
    for v in setdiff(collect(vertices(tn)), my_vertices)
        rem_vertex!(local_tn, v)
    end
    return local_tn
end

# rem_vertex! is only defined for TensorNetwork, so partition the wrapped network.
function local_partition(ψ::TensorNetworkState, my_vertices)
    return TensorNetworkState(
        local_partition(TNQS.tensornetwork(ψ), my_vertices),
        restrict(siteinds(ψ), my_vertices)
    )
end

directed_edges(tn) = reduce(vcat, [[e, reverse(e)] for e in edges(tn)])

# Seed clean deltas rather than running a local update, whose boundary messages carry the dangling
# virtual index of the removed remote neighbour. Cut edges are seeded by the constructor.
function seeded_cache(local_tn)
    bpc = BeliefPropagationCache(local_tn)
    des = directed_edges(local_tn)
    TNQS.setmessages!(bpc, des, [TNQS.default_message(bpc, e) for e in des])
    return bpc
end

# This rank's cut edges, oriented away from it, straight off the case description.
function expected_cut_edges(case)
    out = NamedEdge[]
    for e in edges(case.g)
        case.ranks[src(e)] == case.ranks[dst(e)] && continue
        case.ranks[src(e)] == RANK && push!(out, NamedEdge(src(e) => dst(e)))
        case.ranks[dst(e)] == RANK && push!(out, NamedEdge(dst(e) => src(e)))
    end
    return out
end

function run_case(case)
    g = case.g
    tn = global_network(g)
    my_vertices = case.parts[RANK + 1]
    local_tn = local_partition(tn, my_vertices)
    bpc = seeded_cache(local_tn)
    mpi_bpc = TNQS.BeliefPropagationCacheMPI(bpc, g, case.ranks; comm = COMM)

    check(mpi_bpc isa TNQS.BeliefPropagationCacheMPI, "constructed cache type")
    check(
        Set(collect(vertices(network(mpi_bpc)))) == Set(my_vertices),
        "local network vertices == partition"
    )

    # No vertex is held twice, so the partitions must tile the graph exactly.
    nheld = MPI.Allreduce(length(my_vertices), MPI.SUM, COMM)
    check(nheld == length(collect(vertices(g))), "vertices held $nheld != nv(g)")

    outgoing = expected_cut_edges(case)
    check(
        Set(keys(mpi_bpc.edges_to_send)) == Set(outgoing),
        "edges_to_send == $(Set(outgoing))"
    )
    check(
        Set(keys(mpi_bpc.edges_to_recv)) == Set(reverse.(outgoing)),
        "edges_to_recv == $(Set(reverse.(outgoing)))"
    )
    check(
        Set(keys(mpi_bpc.ghost_ranks)) == Set(dst.(outgoing)),
        "ghost_ranks keys == $(Set(dst.(outgoing)))"
    )
    for e in outgoing
        check(
            mpi_bpc.ghost_ranks[dst(e)] == case.ranks[dst(e)],
            "ghost_ranks[$(dst(e))] == $(case.ranks[dst(e)])"
        )
    end

    # The check on `factor_inds`: both directions must carry exactly the indices a rank holding both
    # endpoints would derive, or a free index is left that grows the message every sweep.
    reference = BeliefPropagationCache(tn)
    for e in [outgoing; reverse.(outgoing)]
        m = TNQS.messages(mpi_bpc)[e]
        check(m isa ITensor, "seeded message on $e is an ITensor")
        want = Set(inds(TNQS.default_message(reference, e)))
        check(
            Set(inds(m)) == want,
            "seeded message on $e has inds $(inds(m)), expected $want"
        )
    end

    # The outgoing cut edges are this rank's to compute; the incoming ones arrive by MPI and must not
    # be in the sequence, or the received message would be overwritten locally.
    seq = TNQS.edge_sequence(mpi_bpc)
    check(all(in(seq), outgoing), "edge_sequence covers the outgoing cut edges")
    check(
        !any(in(seq), reverse.(outgoing)),
        "edge_sequence excludes the incoming cut edges"
    )
    check(
        Set(seq) == Set([directed_edges(local_tn); outgoing]),
        "edge_sequence == local directed edges plus outgoing cut edges"
    )

    serial = update(BeliefPropagationCache(tn); maxiter = case.maxiter, tolerance = nothing)
    mpi_bpc = update(mpi_bpc; maxiter = case.maxiter, tolerance = nothing)

    # The runs use different edge orderings, so they agree only at the fixed point: both
    # thresholds assert convergence. message_diff is a fidelity, so a diff of e bounds derived
    # scalars at sqrt(e) -- keep the pair consistent or non-convergence looks like a bug.
    # Cut edges both ways: the incoming one is the peer's work, so this pins `communicate_messages!`.
    for e in [directed_edges(local_tn); outgoing; reverse.(outgoing)]
        d = TNQS.message_diff(message(mpi_bpc, e), message(serial, e))
        check(d < 1.0e-14, "message $e differs from serial: diff = $d")
    end
    for v in my_vertices
        a, b = TNQS.vertex_scalar(mpi_bpc, v), TNQS.vertex_scalar(serial, v)
        check(isapprox(a, b; rtol = 1.0e-6), "vertex_scalar $v: $a vs serial $b")
    end

    # update_iteration! reduces the diff, so avg_diff is global and a tolerance exit breaks the
    # loop at the same sweep on every rank. Without that the first rank to converge stops
    # calling communicate_messages! and the others block in MPI.recv. A fresh seeded cache is
    # needed because the constructor inserts cut-edge entries into the one it is given.
    tol_bpc = TNQS.BeliefPropagationCacheMPI(
        seeded_cache(local_tn), g, case.ranks; comm = COMM
    )
    tol_bpc = update(tol_bpc; maxiter = case.maxiter, tolerance = 1.0e-12)
    for e in directed_edges(local_tn)
        d = TNQS.message_diff(message(tol_bpc, e), message(mpi_bpc, e))
        check(d < 1.0e-12, "tolerance exit differs from fixed-sweep on $e: diff = $d")
    end

    ghosts = Set(src(e) for e in keys(mpi_bpc.edges_to_recv))
    check(
        Set(collect(vertices(TNQS.messages_graph(mpi_bpc)))) ==
            union(Set(my_vertices), ghosts),
        "messages_graph carries exactly the ghosts $ghosts"
    )

    # A global scalar over the whole partitioned network, so this pins the freenergy reduction --
    # which now counts every vertex once and shares each cut edge between two ranks.
    f_mpi = TNQS.freenergy(mpi_bpc)
    f_serial = TNQS.freenergy(serial)
    check(
        isapprox(f_mpi, f_serial; rtol = 1.0e-6),
        "freenergy = $f_mpi vs serial $f_serial"
    )

    return mpi_bpc
end

# One Trotter layer, sorted so every rank builds an identical gate order.
function trotter_layer(g)
    layer = []
    append!(layer, ("Rx", [v], 0.3) for v in sort(collect(vertices(g))))
    append!(
        layer,
        ("Rzz", [src(e), dst(e)], 0.4) for e in sort(collect(edges(g)); by = repr)
    )
    return layer
end

# expect()'s Algorithm"bp" methods only accept a BeliefPropagationCache, so this repeats their
# body. Valid only for observables supported inside one partition; `vs` must be connected.
function bpexpect(cache, ops::AbstractString, vs)
    incoming = TNQS.incoming_messages(cache, vs)
    function region(op_of)
        ts = TNQS.norm_factors(network(cache), vs; op_strings = op_of)
        append!(ts, incoming)
        return scalar(contract(ts; sequence = TNQS.contraction_sequence(ts; alg = "optimal")))
    end
    opmap = Dict(zip(vs, [string(c) for c in ops]))
    return region(v -> get(opmap, v, "I")) / region(v -> "I")
end

# Runs the same circuit serially on the whole network and distributed over the
# partitions, then compares local observables. The gates are converted against the
# global graph and site indices: the tuple form cannot be used here because each rank
# only holds part of the circuit's support.
function run_apply_case(case)
    ψ = global_state(case.g)
    return check_apply(case, ψ, local_partition(ψ, case.parts[RANK + 1]))
end

# The entry point converts the circuit itself, so this catches a conversion done against the
# rank-local graph, which cannot reach the far vertex of a gate on a cut edge.
function run_entrypoint_apply_case(case)
    g = case.g
    ψ = global_state(g)
    my_vertices = case.parts[RANK + 1]
    bp_update_kwargs = (; maxiter = case.maxiter, tolerance = nothing)
    # Left at its default, unlike `check_apply`, to exercise the singular-value rescaling that both
    # ranks of a cut edge must do identically. Expectation values are ratios, so still comparable.
    apply_kwargs = (; maxdim = 16, cutoff = 0.0)
    layer = trotter_layer(g)

    gates = TNQS.toitensor(layer, g, siteinds(ψ))
    serial = update(BeliefPropagationCache(ψ); bp_update_kwargs...)
    serial, _ = TNQS.apply_gates(
        ITensor[gate[1] for gate in gates], serial;
        gate_vertices = [gate[2] for gate in gates], apply_kwargs, bp_update_kwargs
    )

    ψ_local, errs = TNQS.apply_gates_mpi(
        layer, local_partition(ψ, my_vertices), g, case.ranks;
        comm = COMM, bp_update_kwargs, apply_kwargs
    )

    check(
        Set(collect(vertices(ψ_local))) == Set(my_vertices),
        "apply_gates_mpi returned the partition it was given"
    )
    check(all(<(1.0e-12), errs), "apply_gates_mpi truncated: $(maximum(errs))")

    # It hands back a network rather than a cache, so observables need a fresh one.
    mpi_bpc = TNQS.BeliefPropagationCacheMPI(seeded_cache(ψ_local), g, case.ranks; comm = COMM)
    mpi_bpc = update(mpi_bpc; bp_update_kwargs...)
    for v in my_vertices
        for op in ("Z", "X")
            a, b = bpexpect(mpi_bpc, op, [v]), bpexpect(serial, op, [v])
            check(isapprox(a, b; atol = 1.0e-8), "entry point <$(op)_$v> = $a vs serial $b")
        end
    end
    return mpi_bpc
end

# The memory-limited path: no rank holds the global network. Ranks agree on the boundary by
# drawing every bond index from the broadcast map.
function run_localbuild_apply_case(case)
    g = case.g
    sinds = global_siteinds(g)
    bond_inds = TNQS.shared_bond_inds(g; comm = COMM)

    my_vertices = case.parts[RANK + 1]
    local_ψ = product_partition(g, my_vertices, sinds, bond_inds)
    reference = product_partition(g, collect(vertices(g)), sinds, bond_inds)

    # Without this the comparison below would measure the construction, not the algorithm.
    for v in my_vertices
        d = norm(local_ψ[v] - reference[v])
        check(d < 1.0e-14, "locally built tensor at $v differs from reference by $d")
    end

    return check_apply(case, reference, local_ψ)
end

# Every tensor is overwritten because tensornetworkstate() mints its own bond indices per call,
# which two independent builds could never agree on. All virtual legs come from `bond_inds`.
function product_partition(super_graph, my_vertices, sinds, bond_inds)
    vs = collect(my_vertices)
    ψ = tensornetworkstate(ComplexF64, v -> "↓", subgraph(super_graph, vs), restrict(sinds, vs))
    for v in vs
        TNQS.setindex_preserve!(ψ, state("↓", only(sinds[v])) * one(ComplexF64), v)
    end
    TNQS.insert_partition_virtualinds!(ψ, super_graph, bond_inds)
    return ψ
end

# Gates are converted against the global graph and site indices: the tuple form cannot be used
# because each rank holds only part of the circuit's support.
function check_apply(case, ψ_reference, local_ψ)
    g = case.g
    my_vertices = case.parts[RANK + 1]
    bp_update_kwargs = (; maxiter = case.maxiter, tolerance = nothing)
    # cutoff 0 with maxdim headroom: no truncation, so both runs do identical arithmetic and
    # any mismatch is a protocol bug rather than lost precision.
    apply_kwargs = (; maxdim = 16, cutoff = 0.0, normalize_tensors = false)

    gates = TNQS.toitensor(trotter_layer(g), g, siteinds(ψ_reference))
    itensors = [gate[1] for gate in gates]
    gate_vertices = [gate[2] for gate in gates]

    serial = update(BeliefPropagationCache(ψ_reference); bp_update_kwargs...)
    serial, serial_errs = TNQS.apply_gates(
        itensors, serial; gate_vertices, apply_kwargs, bp_update_kwargs
    )

    mpi_bpc = TNQS.BeliefPropagationCacheMPI(
        seeded_cache(local_ψ), g, case.ranks; comm = COMM
    )
    # Converge first so the distributed run starts from the same messages as serial.
    mpi_bpc = update(mpi_bpc; bp_update_kwargs...)

    # A boundary gate is taken up by BOTH its ranks but computed by exactly one; disagreement there
    # is a deadlock rather than a wrong answer.
    roles = [
        TNQS.gate_role(gv, my_vertices, mpi_bpc.ghost_ranks) for gv in gate_vertices
    ]
    participants = MPI.Allreduce([r.kind === :skip ? 0 : 1 for r in roles], MPI.SUM, COMM)
    computers = MPI.Allreduce(
        [(r.kind !== :skip && r.compute) ? 1 : 0 for r in roles], MPI.SUM, COMM
    )
    expected = [length(unique(case.ranks[v] for v in gv)) for gv in gate_vertices]
    check(participants == expected, "participants per gate $participants != $expected")
    check(all(==(1), computers), "each gate computed once, got $computers")
    check(
        all(i -> roles[i].kind === :boundary ? expected[i] == 2 : true, eachindex(roles)),
        ":boundary claimed for a gate inside one rank"
    )

    mpi_bpc, mpi_errs = TNQS.apply_gates(
        itensors, mpi_bpc; gate_vertices, apply_kwargs, bp_update_kwargs
    )

    check(mpi_bpc isa TNQS.BeliefPropagationCacheMPI, "cache survives apply_gates")
    check(
        Set(collect(vertices(network(mpi_bpc)))) == Set(my_vertices),
        "apply_gates did not change the local partition"
    )
    check(all(<(1.0e-12), serial_errs), "serial run truncated: $(maximum(serial_errs))")
    check(all(<(1.0e-12), mpi_errs), "distributed run truncated: $(maximum(mpi_errs))")

    # A tensor that came back from the boundary protocol with the wrong bond index would contract
    # into an outer product rather than a bond, which shows up as extra free indices.
    for v in my_vertices
        t = network(mpi_bpc)[v]
        check(all(i -> plev(i) == 0, inds(t)), "tensor at $v kept a primed index")
        nvirtual = length(collect(inds(t))) - length(siteinds(network(mpi_bpc))[v])
        check(
            nvirtual == length(neighbors(g, v)),
            "tensor at $v has $nvirtual virtual legs, expected $(length(neighbors(g, v)))"
        )
    end

    for v in my_vertices
        for op in ("Z", "X")
            a, b = bpexpect(mpi_bpc, op, [v]), bpexpect(serial, op, [v])
            check(isapprox(a, b; atol = 1.0e-8), "<$(op)_$v> = $a vs serial $b")
        end
    end
    for e in edges(network(mpi_bpc))
        vs = [src(e), dst(e)]
        a, b = bpexpect(mpi_bpc, "ZZ", vs), bpexpect(serial, "ZZ", vs)
        check(isapprox(a, b; atol = 1.0e-8), "<Z_$(src(e)) Z_$(dst(e))> = $a vs serial $b")
    end

    # A global scalar, so this also pins the freenergy reduction.
    ip = TNQS.inner_mpi(
        network(mpi_bpc), network(mpi_bpc), g, case.ranks; comm = COMM, bp_update_kwargs
    )
    ip_serial = TNQS.inner(
        network(serial), network(serial); alg = "bp", cache_update_kwargs = bp_update_kwargs
    )
    check(isapprox(ip, ip_serial; rtol = 1.0e-6), "inner_mpi = $ip vs serial $ip_serial")

    return mpi_bpc
end

const CASES = Dict(
    "path" => (run_case, path_case),
    "ring" => (run_case, ring_case),
    "chain3" => (run_case, chain3_case),
    "comb" => (run_case, comb_case),
    "apply_path" => (run_apply_case, apply_path_case),
    "apply_ring" => (run_apply_case, apply_ring_case),
    "apply_comb" => (run_apply_case, apply_comb_case),
    "entry_path" => (run_entrypoint_apply_case, apply_path_case),
    "entry_comb" => (run_entrypoint_apply_case, apply_comb_case),
    "localbuild_path" => (run_localbuild_apply_case, apply_path_case),
    "localbuild_ring" => (run_localbuild_apply_case, apply_ring_case)
)

const CASE_NAME = ARGS[1]
let (runner, case_fn) = CASES[CASE_NAME]
    runner(case_fn())
end

# Agree on the verdict so every rank exits the same way and none is left blocking.
const TOTAL = MPI.Allreduce(length(FAILURES), MPI.SUM, COMM)
for f in FAILURES
    println(stderr, f)
end
RANK == 0 && println("case $CASE_NAME: $TOTAL failure(s)")
MPI.Finalize()
exit(TOTAL == 0 ? 0 : 1)
