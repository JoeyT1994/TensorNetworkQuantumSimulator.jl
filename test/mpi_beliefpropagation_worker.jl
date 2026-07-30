# Runs under mpiexec. One positional arg: a case name from CASES below.
# Deliberately avoids Test and Random, which are in [extras] and so unavailable
# under --project=<package>.
using Dictionaries: Dictionary, delete!
using Graphs: dst, neighbors, src
using ITensors: ITensors, ITensor, contract, scalar, state
using LinearAlgebra: norm
using MPI
using NamedGraphs.GraphsExtensions: subgraph
using NamedGraphs: NamedEdge, rem_vertex!
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)

# BP on a loopy partition does not reach a fixed point for every random draw, and the cases
# below compare against serial to 1e-14 -- which only means anything at the fixed point. Left
# on the global RNG the loopy cases fail every so often, and a flaky case is indistinguishable
# from a protocol bug. Random.seed! is not an option (Random lives in [extras], so it is absent
# under --project=<package>), hence this small LCG.
mutable struct Lcg
    state::UInt64
end
function next!(rng::Lcg)
    rng.state = rng.state * 0x5851f42d4c957f2d + 0x14057b7ef767814f
    return Float64(rng.state >> 11) / Float64(UInt64(1) << 53) - 0.5
end

# Overwrites the storage in place, so the index structure the constructors chose is untouched.
# Only ever called on the rank that then broadcasts the network, so cross-rank agreement still
# comes from the broadcast.
function fill_deterministic!(tn, seed::Integer)
    rng = Lcg(UInt64(seed))
    for v in sort(collect(vertices(tn)))
        d = ITensors.data(tn[v])
        for i in eachindex(d)
            d[i] = eltype(d) <: Complex ? complex(next!(rng), next!(rng)) : next!(rng)
        end
    end
    return tn
end

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

apply_path_case() = (; path_case()..., name = "6-site path apply, 2 ranks", maxiter = 25)
apply_ring_case() = (; ring_case()..., name = "6-site ring apply, 2 ranks", maxiter = 100)

# All ranks must share byte-identical Index objects, so the network is built once
# and broadcast; Random.seed! does not reproduce ITensor Index ids.
function global_network(g; seed = 20250729)
    tn = RANK == 0 ?
        fill_deterministic!(random_tensornetwork(Float64, g; bond_dimension = 2), seed) :
        nothing
    return MPI.bcast(tn, COMM; root = 0)
end

function global_state(g; seed = 11071988)
    ψ = RANK == 0 ?
        fill_deterministic!(random_tensornetworkstate(ComplexF64, g; bond_dimension = 2), seed) :
        nothing
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
    check(
        Set(keys(mpi_bpc.edges_to_send)) == expected_send,
        "edges_to_send == $expected_send"
    )
    check(
        Set(keys(mpi_bpc.edges_to_recv)) == expected_recv,
        "edges_to_recv == $expected_recv"
    )

    # Received messages must be present and carry the boundary tensor's dangling index.
    for e in expected_recv
        m = messages(mpi_bpc)[e]
        check(m isa ITensor, "received message on $e is an ITensor")
    end

    serial = update(BeliefPropagationCache(tn); maxiter = case.maxiter, tolerance = nothing)
    mpi_bpc = update(mpi_bpc; maxiter = case.maxiter, tolerance = nothing)

    # Everything below compares the two runs at their fixed point, which is only meaningful if
    # they got there. Asserted separately so that a network BP simply cannot solve reports
    # itself rather than showing up as a pile of mismatches.
    serial_extra = update(serial; maxiter = 1, tolerance = nothing)
    for e in directed_edges(tn)
        d = TNQS.message_diff(message(serial_extra, e), message(serial, e))
        check(d < 1.0e-14, "serial run is not at a fixed point on $e (diff = $d); bad seed?")
    end

    # The runs use different edge orderings, so they agree only at the fixed point: both
    # thresholds assert convergence. message_diff is a fidelity, so a diff of e bounds derived
    # scalars at sqrt(e) -- keep the pair consistent or non-convergence looks like a bug.
    for e in directed_edges(local_tn)
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
    # needed because the constructor inserts ghost entries into the one it is given.
    tol_bpc = TNQS.BeliefPropagationCacheMPI(
        seeded_cache(local_tn), g, case.shared; comm = COMM
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

    # apply_gates reads keys() to decide which factors to exchange, so a stray key would make
    # a rank wait on a factor nobody sends.
    held = Set(v for (v, ranks) in pairs(case.shared) if RANK in ranks)
    check(Set(keys(mpi_bpc.shared_vertices)) == held, "shared_vertices keys == $held")
    for v in held
        r1, r2 = case.shared[v]
        other = RANK == r1 ? r2 : r1
        check(mpi_bpc.shared_vertices[v] == other, "shared_vertices[$v] == $other")
    end

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
        seeded_cache(local_ψ), g, case.shared; comm = COMM
    )
    # Converge first so the distributed run starts from the same messages as serial.
    mpi_bpc = update(mpi_bpc; bp_update_kwargs...)
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

    # A rank applies a gate exactly when it holds all its vertices: one rank for a gate
    # crossing a boundary, both holders for a one-site gate on a shared vertex.
    applied = [
        first(TNQS.should_apply_gate(gv, my_vertices, mpi_bpc.shared_vertices)) ? 1 : 0
            for gv in gate_vertices
    ]
    holders = [all(in(my_vertices), gv) ? 1 : 0 for gv in gate_vertices]
    owners = MPI.Allreduce(applied, MPI.SUM, COMM)
    nholders = MPI.Allreduce(holders, MPI.SUM, COMM)
    check(all(>=(1), owners), "every gate applied by some rank, got $owners")
    check(owners == nholders, "owners $owners != ranks holding all gate vertices $nholders")

    # Shared vertices are included, so a factor that failed to cross a boundary shows up here.
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
        network(mpi_bpc), network(mpi_bpc), g, case.shared; comm = COMM, bp_update_kwargs
    )
    ip_serial = TNQS.inner(
        network(serial), network(serial); alg = "bp", cache_update_kwargs = bp_update_kwargs
    )
    check(isapprox(ip, ip_serial; rtol = 1.0e-6), "inner_mpi = $ip vs serial $ip_serial")

    return mpi_bpc
end

# The top-level entry point, driven exactly as a user would: a tuple circuit spanning the whole
# graph, no hand-conversion to ITensors and no explicit gate_vertices. This is the path that
# needs `super_graph` to resolve gate supports and lazy per-rank `toitensor` conversion, since
# no rank holds the site indices of the whole circuit.
function run_apply_gates_mpi_case(case)
    g = case.g
    ψ = global_state(g)
    my_vertices = case.parts[RANK + 1]
    bp_update_kwargs = (; maxiter = case.maxiter, tolerance = nothing)
    apply_kwargs = (; maxdim = 16, cutoff = 0.0, normalize_tensors = false)
    circuit = trotter_layer(g)

    serial, serial_errs = TNQS.apply_gates(
        circuit, ψ; apply_kwargs, bp_update_kwargs
    )

    local_ψ, errs = TNQS.apply_gates_mpi(
        circuit, local_partition(ψ, my_vertices), g, case.shared;
        comm = COMM, bp_update_kwargs, apply_kwargs
    )

    check(local_ψ isa TensorNetworkState, "apply_gates_mpi returns a state")
    check(
        Set(collect(vertices(local_ψ))) == Set(my_vertices),
        "apply_gates_mpi did not change the local partition"
    )
    # Reduced across ranks, so every rank sees every gate's error rather than 0 for the gates it
    # skipped.
    check(
        isapprox(errs, serial_errs; atol = 1.0e-10),
        "truncation errors not reduced: max |Δ| = $(maximum(abs.(errs - serial_errs)))"
    )

    mpi_bpc = update(
        TNQS.BeliefPropagationCacheMPI(seeded_cache(local_ψ), g, case.shared; comm = COMM);
        bp_update_kwargs...
    )
    serial_bpc = update(BeliefPropagationCache(serial); bp_update_kwargs...)
    for v in my_vertices
        for op in ("Z", "X")
            # expect() on the distributed cache, rather than the hand-rolled bpexpect above.
            a = expect(mpi_bpc, (op, [v]))
            b = expect(serial_bpc, (op, [v]))
            check(isapprox(a, b; atol = 1.0e-8), "<$(op)_$v> = $a vs serial $b")
        end
    end
    for e in edges(network(mpi_bpc))
        vs = [src(e), dst(e)]
        a, b = expect(mpi_bpc, ("ZZ", vs)), expect(serial_bpc, ("ZZ", vs))
        check(isapprox(a, b; atol = 1.0e-8), "<Z_$(src(e)) Z_$(dst(e))> = $a vs serial $b")
    end

    # An observable straddling a boundary cannot be measured from one partition, and must say so
    # rather than quietly answering from a truncated region.
    remote = first(setdiff(collect(vertices(g)), my_vertices))
    threw = try
        expect(mpi_bpc, ("Z", [remote]))
        false
    catch
        true
    end
    check(threw, "expect on the non-local vertex $remote should error")

    return mpi_bpc
end

# Rank-dependent kwargs used to deadlock: `maxiter` defaulted to `is_tree(local partition)`, so a
# rank whose piece is a tree ran one sweep and left the rest blocking in MPI.Recv. Passing no
# kwargs at all now has to terminate and agree with a generously-converged serial run.
function run_default_kwargs_case(case)
    g = case.g
    tn = global_network(g)
    my_vertices = case.parts[RANK + 1]
    local_tn = local_partition(tn, my_vertices)

    mpi_bpc = TNQS.BeliefPropagationCacheMPI(
        seeded_cache(local_tn), g, case.shared; comm = COMM
    )
    kwargs = TNQS.default_bp_update_kwargs(mpi_bpc)
    check(
        kwargs.maxiter >= MPI.Comm_size(COMM),
        "default maxiter $(kwargs.maxiter) leaves no room to cross $(MPI.Comm_size(COMM)) partitions"
    )
    # Identical on every rank, or the run cannot terminate together.
    lo = MPI.Allreduce(kwargs.maxiter, MPI.MIN, COMM)
    hi = MPI.Allreduce(kwargs.maxiter, MPI.MAX, COMM)
    check(lo == hi, "default maxiter differs across ranks: $lo vs $hi")

    # The point of this case is that the run terminates on every rank with no kwargs at all. The
    # comparison is deliberately loose: the default tolerance is on message_diff, a fidelity, so
    # it only pins derived scalars to its square root (1e-8 -> ~1e-4).
    mpi_bpc = update(mpi_bpc)
    serial = update(BeliefPropagationCache(tn); maxiter = 400, tolerance = 1.0e-14)
    for v in my_vertices
        a, b = TNQS.vertex_scalar(mpi_bpc, v), TNQS.vertex_scalar(serial, v)
        check(isapprox(a, b; rtol = 1.0e-3), "default-kwargs vertex_scalar $v: $a vs $b")
    end
    return mpi_bpc
end

# Malformed partitions must throw on every rank instead of hanging on a mismatched exchange.
function run_validation_case(case)
    g = named_grid((6, 1))
    my_vertices = case.parts[RANK + 1]
    tn = global_network(g)

    function rejects(name, parts, shared)
        local_tn = local_partition(tn, parts[RANK + 1])
        threw = try
            TNQS.BeliefPropagationCacheMPI(seeded_cache(local_tn), g, shared; comm = COMM)
            false
        catch
            true
        end
        # Reduced, so a rank that failed to notice is a failure too: if only some ranks throw,
        # the others are left in a collective the first will never join.
        agreed = MPI.Allreduce(threw ? 1 : 0, MPI.SUM, COMM) == MPI.Comm_size(COMM)
        return check(agreed, "$name should be rejected on every rank")
    end

    # Adjacent shared vertices: both holders would own the edge between them.
    rejects(
        "adjacent shared vertices",
        [[(i, 1) for i in 1:4], [(3, 1), (4, 1), (5, 1), (6, 1)]],
        Dictionary([(3, 1), (4, 1)], [(Int32(0), Int32(1)), (Int32(0), Int32(1))])
    )
    # A vertex claimed by three ranks.
    rejects(
        "three-way shared vertex",
        [[(i, 1) for i in 1:3], [(i, 1) for i in 3:6]],
        Dictionary([(3, 1)], [(Int32(0), Int32(1), Int32(1))])
    )
    # An out-of-range rank.
    rejects(
        "rank outside the communicator",
        [[(i, 1) for i in 1:3], [(i, 1) for i in 3:6]],
        Dictionary([(3, 1)], [(Int32(0), Int32(7))])
    )
    # A partition that drops an edge of the super graph.
    rejects(
        "uncovered edge",
        [[(i, 1) for i in 1:3], [(i, 1) for i in 4:6]],
        Dictionary([(3, 1)], [(Int32(0), Int32(1))])
    )
    return nothing
end

# The fallback taken when the tensors are on a GPU but MPI cannot read device memory: payloads
# are mirrored through host buffers. Without a GPU the mirrors are host arrays too, but the offset
# and copy bookkeeping -- and the disabling of the single-tensor zero-copy path -- are the same
# code, so this still guards the part that a segfault deep inside MPI would otherwise be the
# first sign of.
function run_host_staging_case(case)
    check_no_scalar_indexing()
    TNQS._FORCE_HOST_STAGING[] = true
    try
        mpi_bpc = run_case(case)
        check(
            TNQS._needs_host_staging(mpi_bpc), "host staging is actually engaged for this case"
        )
        return mpi_bpc
    finally
        TNQS._FORCE_HOST_STAGING[] = false
    end
end

# Stands in for a device array in the one respect that broke: scalar indexing throws, so any copy
# that falls back to Base's elementwise loop fails here exactly as it does on a CuArray. Only the
# strided five-argument `copyto!` is provided, which is what a real device array offers.
struct NoScalarArray{T} <: AbstractArray{T, 1}
    data::Vector{T}
end
Base.size(a::NoScalarArray) = size(a.data)
Base.getindex(::NoScalarArray, ::Int) = error("scalar indexing is disallowed")
Base.setindex!(::NoScalarArray, ::Any, ::Int) = error("scalar indexing is disallowed")
function Base.similar(::NoScalarArray, ::Type{T}, dims::Base.Dims{1}) where {T}
    return NoScalarArray(Vector{T}(undef, dims[1]))
end
function Base.copyto!(d::NoScalarArray, do_::Integer, s::NoScalarArray, so::Integer, n::Integer)
    copyto!(d.data, do_, s.data, so, n)
    return d
end
function Base.copyto!(d::NoScalarArray, do_::Integer, s::Array, so::Integer, n::Integer)
    copyto!(d.data, do_, s, so, n)
    return d
end
function Base.copyto!(d::Array, do_::Integer, s::NoScalarArray, so::Integer, n::Integer)
    copyto!(d, do_, s.data, so, n)
    return d
end

# Guards the host<->device staging copies directly, without needing a GPU: every one of them has
# to survive a source or destination that refuses scalar access, and land at the right offset.
function check_no_scalar_indexing()
    src = NoScalarArray(collect(1.0:8.0))
    host = fill(-1.0, 8)
    ok = try
        TNQS._copy_range!(host, 1, src, 1, 8)
        true
    catch e
        check(false, "device->host staging copy threw: $e")
        false
    end
    ok && check(host == collect(1.0:8.0), "device->host staging copy moved the wrong data")

    back = NoScalarArray(fill(-1.0, 8))
    ok = try
        TNQS._copy_range!(back, 1, collect(1.0:8.0), 1, 8)
        true
    catch e
        check(false, "host->device staging copy threw: $e")
        false
    end
    ok && check(back.data == collect(1.0:8.0), "host->device staging copy moved the wrong data")

    # A batch with more than one tensor packs at an offset, which is where an off-by-one hides.
    offset_dest = fill(-1.0, 8)
    TNQS._copy_range!(offset_dest, 5, NoScalarArray(collect(1.0:4.0)), 1, 4)
    check(
        offset_dest == [-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, 3.0, 4.0],
        "offset staging copy landed in the wrong place: $offset_dest"
    )
    return nothing
end

# `blocked_gates!(true)` is a process-global switch, and `apply_gates_mpi` reaches
# `simple_update` through the same `apply_gate!` as the serial path, so flipping it is all that is
# needed to route a distributed run through the memory-bounded gate. This asserts that: the
# distributed result with the switch on must still match the serial reference with it off.
function run_blocked_apply_case(case)
    g = case.g
    ψ = global_state(g)
    my_vertices = case.parts[RANK + 1]
    bp_update_kwargs = (; maxiter = case.maxiter, tolerance = nothing)
    apply_kwargs = (; maxdim = 16, cutoff = 0.0, normalize_tensors = false)
    circuit = trotter_layer(g)

    # Reference: serial, standard gate path.
    serial, serial_errs = TNQS.apply_gates(circuit, ψ; apply_kwargs, bp_update_kwargs)

    blocked_gates!(true)
    local_ψ, errs = try
        TNQS.apply_gates_mpi(
            circuit, local_partition(ψ, my_vertices), g, case.shared;
            comm = COMM, bp_update_kwargs, apply_kwargs
        )
    finally
        blocked_gates!(false)
    end
    check(!blocked_gates(), "the switch is restored afterwards")
    check(
        isapprox(errs, serial_errs; atol = 1.0e-10),
        "blocked distributed truncation errors match serial: max |Δ| = " *
            "$(maximum(abs.(errs - serial_errs)))"
    )

    mpi_bpc = update(
        TNQS.BeliefPropagationCacheMPI(seeded_cache(local_ψ), g, case.shared; comm = COMM);
        bp_update_kwargs...
    )
    serial_bpc = update(BeliefPropagationCache(serial); bp_update_kwargs...)
    for v in my_vertices, op in ("Z", "X")
        a, b = expect(mpi_bpc, (op, [v])), expect(serial_bpc, (op, [v]))
        check(isapprox(a, b; atol = 1.0e-8), "blocked <$(op)_$v> = $a vs serial $b")
    end
    return mpi_bpc
end

const CASES = Dict(
    "path" => (run_case, path_case),
    "ring" => (run_case, ring_case),
    "chain3" => (run_case, chain3_case),
    "apply_path" => (run_apply_case, apply_path_case),
    "apply_ring" => (run_apply_case, apply_ring_case),
    "localbuild_path" => (run_localbuild_apply_case, apply_path_case),
    "localbuild_ring" => (run_localbuild_apply_case, apply_ring_case),
    "apply_gates_mpi_path" => (run_apply_gates_mpi_case, apply_path_case),
    "apply_gates_mpi_ring" => (run_apply_gates_mpi_case, apply_ring_case),
    "defaults_path" => (run_default_kwargs_case, path_case),
    "defaults_ring" => (run_default_kwargs_case, ring_case),
    "defaults_chain3" => (run_default_kwargs_case, chain3_case),
    "validation" => (run_validation_case, path_case),
    "host_staging_path" => (run_host_staging_case, path_case),
    "host_staging_ring" => (run_host_staging_case, ring_case),
    "blocked_apply_path" => (run_blocked_apply_case, apply_path_case),
    "blocked_apply_ring" => (run_blocked_apply_case, apply_ring_case)
)

# Every case named on the command line runs in this one process. Loading the package and
# JIT-compiling the BP paths costs far more than any single case, so running the cases of a given
# rank count together rather than one mpiexec each is most of the suite's wall time.
for case_name in ARGS
    haskey(CASES, case_name) || error("unknown case $case_name; have $(sort(collect(keys(CASES))))")
    before = length(FAILURES)
    let (runner, case_fn) = CASES[case_name]
        runner(case_fn())
    end
    # Reduced per case so the report says which one failed, not just that something did.
    n = MPI.Allreduce(length(FAILURES) - before, MPI.SUM, COMM)
    RANK == 0 && println("case $case_name: $n failure(s)")
    flush(stdout)
end

# Agree on the verdict so every rank exits the same way and none is left blocking.
const TOTAL = MPI.Allreduce(length(FAILURES), MPI.SUM, COMM)
for f in FAILURES
    println(stderr, f)
end
RANK == 0 && println("total: $TOTAL failure(s)")
MPI.Finalize()
exit(TOTAL == 0 ? 0 : 1)
