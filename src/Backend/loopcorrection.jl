# the main loop correctio function
function loop_correction_factor(bpc::BeliefPropagationCache, max_circuit_size::Int; max_genus=2)
    circuits = enumerate_circuits(bpc, max_circuit_size; max_genus)
    return loop_correction_factor(bpc, circuits)
end

function loop_correction_factor(bpc::BeliefPropagationCache, circuits)
    if isempty(circuits)
        return 1
    end

    return 1 + sum(circuit_weights(bpc, circuits))
end

# finding all the loops in the network
function enumerate_cycles(bpc::BeliefPropagationCache, args...)
    return enumerate_cycles(partitioned_tensornetwork(bpc), args...)
end

function enumerate_cycles(pg::PartitionedGraph, args...)
    return [PartitionEdge.(cycle) for cycle in enumerate_cycles(partitioned_graph(pg), args...)]
end

function enumerate_cycles(g::NamedGraph, max_cycle_length::Int64)
    vs = collect(vertices(g))
    vertex_cycles = simplecycles_limited_length(position_graph(g), max_cycle_length)
    vertex_cycles = [[vs[i] for i in cycle] for cycle in vertex_cycles]
    cycles = vertex_cycle_to_cycle.(vertex_cycles)
    cycles = filter(c -> is_valid_cycle(g, c), cycles)
    return unique(cycles, edgevectors_equal)
end

function enumerate_circuits(bpc::BeliefPropagationCache, args...; kwargs...)
    return enumerate_circuits(partitioned_tensornetwork(bpc), args...; kwargs...)
end

function enumerate_circuits(pg::PartitionedGraph, args...; kwargs...)
    # TODO: why is this recursive?
    return [PartitionEdge.(circuit) for circuit in enumerate_circuits(partitioned_graph(pg), args...; kwargs...)]
end

function vertices_in_edgevector(edges::Vector)
    return unique(vcat(src.(edges), dst.(edges)))
end

function sim_src(bpc::BeliefPropagationCache, pe::PartitionEdge)
    bpc = copy(bpc)
    mer = only(message(bpc, reverse(pe)))
    linds = inds(mer)
    linds_sim = sim.(linds)
    mer = replaceinds(mer, inds(mer), linds_sim)
    ms = messages(bpc)
    set!(ms, reverse(pe), ITensor[mer])
    verts = vertices(bpc, src(pe))
    for v in verts
        t = only(factors(bpc, [v]))
        t_inds = filter(i -> i ∈ linds, inds(t))
        if !isempty(t_inds)
            t_ind = only(t_inds)
            t_ind_pos = findfirst(x -> x == t_ind, linds)
            t = replaceind(t, t_ind, linds_sim[t_ind_pos])
            bpc = update_factor(bpc, v, t)
        end
    end
    return bpc
end

function sim_srcs(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
    bpc = copy(bpc)
    for pe in pes
        bpc = sim_src(bpc, pe)
    end
    return bpc
end

function sim_partitionvertex(bpc::BeliefPropagationCache, pv::PartitionVertex)
    return sim_srcs(bpc, boundary_partitionedges(bpc, [pv]; dir=:out))
end

function sim_circuit(bpc::BeliefPropagationCache, circuit)
    bpc = copy(bpc)
    pvs = vertices_in_edgevector(circuit)
    for pv in pvs
        bpc = sim_partitionvertex(bpc, pv)
    end
    return bpc
end

function ITensorNetworks.boundary_partitionedges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
    pvs = vertices_in_edgevector(pes)
    bpes = PartitionEdge[]
    for pv in pvs
        incoming_es = boundary_partitionedges(bpc, pv; dir=:in)
        incoming_es = filter(e -> e ∉ pes && reverse(e) ∉ pes, incoming_es)
        append!(bpes, incoming_es)
    end
    return bpes
end

function antiprojector(bpc::BeliefPropagationCache, pe::PartitionEdge)
    me, mer = only(message(bpc, pe)), only(message(bpc, reverse(pe)))
    me_inds, mer_inds = [i for i in inds(me)], [i for i in inds(mer)]
    row_inds, col_inds = sort(me_inds; by=ind -> plev(ind)), sort(mer_inds; by=ind -> plev(ind))
    row_combiner, col_combiner = combiner(row_inds), combiner(col_inds)
    ap = delta(combinedind(row_combiner), combinedind(col_combiner)) * row_combiner * col_combiner
    return ap - me * mer
end

#Return weight of term with anti projectors on all edges in the circuit and messages everywhere else
function circuit_weight(bpc::BeliefPropagationCache, circuit)
    bpc = sim_circuit(bpc, circuit)
    incoming_ms = ITensor[only(message(bpc, pe)) for pe in boundary_partitionedges(bpc, circuit)]
    local_tensors = factors(bpc, vertices(bpc, vertices_in_edgevector(circuit)))
    antiprojectors = ITensor[antiprojector(bpc, pe) for pe in circuit]
    ts = [incoming_ms; local_tensors; antiprojectors]
    seq = contraction_sequence(ts; alg="einexpr", optimizer=Greedy())
    return contract(ts; sequence=seq)[]
end

function circuit_weights(bpc, circuits, args...)
    return [circuit_weight(bpc, circuit, args...) for circuit in circuits]
end

function cycles_to_circuit(cycles::Vector)
    circuit = reduce(vcat, cycles)
    true_circuit = []
    for e in circuit
        if reverse(e) ∉ true_circuit && e ∉ true_circuit
            push!(true_circuit, e)
        end
    end
    return true_circuit
end

function circuits_from_cycles(cycles::Vector, max_circuit_size::Int64; max_genus::Int64=2)
    circuit_components = collect(powerset(cycles, 1, max_genus))
    circuits = [cycles_to_circuit(cycles) for cycles in circuit_components]
    circuits = unique(circuits, edgevectors_equal)
    return filter(l -> length(l) <= max_circuit_size, circuits)
end

function enumerate_circuits(g::NamedGraph, max_circuit_size::Int64; max_genus::Int64=2)
    cycles = enumerate_cycles(g, max_circuit_size)
    return circuits_from_cycles(cycles, max_circuit_size; max_genus)
end

function vertex_cycle_to_cycle(cycle::Vector)
    return vcat([NamedEdge(cycle[i] => cycle[i+1]) for i in 1:(length(cycle)-1)], [NamedEdge(first(cycle) => last(cycle))])
end

function is_valid_cycle(g::AbstractGraph, cycle::Vector)
    length(cycle) == 2 && return false
    for e in cycle
        !has_edge(g, e) && return false
    end
    return true
end

function edgevectors_equal(edges1::Vector, edges2::Vector)
    length(edges1) != length(edges2) && return false
    for e in edges1
        if e ∉ edges2 && reverse(e) ∉ edges2
            return false
        end
    end
    return true
end

function Base.unique(objects::Vector, equality_function::Function)
    unique_objects = []
    for object1 in objects
        if !any([equality_function(object1, object2) for object2 in unique_objects])
            push!(unique_objects, object1)
        end
    end
    return unique_objects
end