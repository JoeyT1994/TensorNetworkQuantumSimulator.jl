using TensorNetworkQuantumSimulator
using NamedGraphs: unique_simplecycles_limited_length
using ITensors: Index, sim

using Graphs: topological_sort
using Graphs.SimpleGraphs: SimpleDiGraph

function special_multiply(t1::ITensor, t2::ITensor)
    cinds = commoninds(t1, t2)
    ds = []
    for cind in cinds
        sim_cind1, sim_cind2 = sim(cind), sim(cind)
        t1 = replaceind(t1, cind, sim_cind1)
        t2 = replaceind(t2, cind, sim_cind2)
        push!(ds, delta([cind, sim_cind1, sim_cind2]))
    end

    t = reduce(*, [[t1, t2 ]; ds])
    return t    
end

#Element wise multiplication of all tensors and sum over specified indices. For efficient message updating.
function hyper_multiply(ts::Vector{<:ITensor}, inds_to_sum_over =[])
    all_inds = reduce(vcat, [inds(t) for t in ts])
    unique_inds = unique(all_inds)
    index_counts = [count(i -> i == ui, all_inds) for ui in unique_inds]

    #Any index that appears more than wise. Sim it amongst all tensors. 
    #If not being summed over, add in a copy with an index, if it is add in a hyper tensor without one.

    for (i, ui) in enumerate(unique_inds)
        if index_counts[i] > 1
            sim_inds = [sim(ui, j) for j in 1:index_counts[i]]
            cnt = 1
            for (j, t) in enumerate(ts)
                if ui ∈ inds(t)
                    t = replaceind(t, ui, sim_inds[cnt])
                    ts[j] = t
                    cnt += 1
                end
            end

            hyper_tensor = ITensor(1.0)
            for si in sim_inds
                hyper_tensor = hyper_tensor * delta(si)
            end
            if ui ∈ inds_to_sum_over
                hyper_tensor *= delta(ui)
            end
            push!(ts, hyper_tensor)
            
        end
    end

    seq = contraction_sequence(ts; alg = "optimal")
    return contract(ts; sequence = seq)
end

function elementwise_operation(f::Function, t::ITensor)
    new_t = copy(t)
    for i in eachindval(t)
        new_t[i...] = f(t[i...])
    end
    return new_t
end

function pointwise_division_raise(a::ITensor, b::ITensor; power = 1)
    @assert Set(inds(a)) == Set(inds(b))
    out = ITensor(eltype(a), 1.0, inds(a))
    indexes = inds(a)
    for iv in eachindval(out)
        out[iv...] = (a[iv...] / b[iv...])^(power)
    end

    return out
end

#All factors and their 4 variables (edges) form the parent regions for simple BP
function construct_bp_bs(T::BeliefPropagationCache)
    g = graph(T)
    es = edges(g)

    regions = Set[]
    for v in vertices(g)
        region = Set{Any}([v])
        for vn in neighbors(g, v)
            e = NamedEdge(v => vn) ∈ es ? NamedEdge(v => vn) : NamedEdge(vn => v)
            @assert e ∈ es
            push!(region, NamedEdge(e))
        end
        push!(regions, region)
    end
    return regions
end

#Here we take all factors and their 4 variables (edges), plus all loops of variables only to form the parent regions for GBP
function construct_gbp_bs(T::BeliefPropagationCache, loop_length::Int)
    g = graph(T)
    g_edges = edges(g)
    bs = construct_bp_bs(T)
    cycles = unique_simplecycles_limited_length(g, loop_length)
    gbp_bs = copy(bs)
    for cycle in cycles
        region = Set()
        for (i, v) in enumerate(cycle)
            e = i != length(cycle) ? NamedEdge(v => cycle[i+1]) : NamedEdge(v => cycle[1])
            e = e ∈ g_edges ? e : reverse(e)
            @assert e ∈ g_edges
            push!(region, e)
        end
        push!(gbp_bs, region)  # Add first vertex to close the loop
    end

    return gbp_bs
end

function intersections(ms)
    intersects = []
    for i in 1:length(ms), j in i+1:length(ms)
        s = intersect(ms[i], ms[j])
        if !isempty(s) && s ∉ intersects
            push!(intersects, s)
        end
    end
    return collect.(intersects)
end

function construct_ms(bs)
    current_ms = intersections(bs)
    all_ms = []

    while !isempty(current_ms)
        for m in current_ms
            if m ∉ all_ms
                push!(all_ms,m)
            end
        end
        current_ms = intersections(current_ms)
    end

    return collect.(all_ms)
end

function parents(m, bs)
    parents = []
    for (i, b) in enumerate(bs)
        if issubset(m, b)
            push!(parents, i)
        end
    end
    return parents
end

function all_parents(ms, bs)
    ms_parents = []
    for m in ms
        push!(ms_parents, parents(m, bs))
    end
    return ms_parents
end

function mobius_numbers(ms, ps)
    #First get the subset matrix
    mat = zeros(Int, length(ms), length(ms))
    for (i, m1) in enumerate(ms), (j, m2) in enumerate(ms[(i + 1):end])
        if issubset(m1, m2)
            mat[i, j + i] = 1
        end
        if issubset(m2, m1)
            mat[j + i, i] = 1
        end
    end

    g = SimpleDiGraph(mat)
    ts = topological_sort(g)
    ts = reverse(ts)
    
    mobius_numbers = zeros(Int, length(ms))
    for i in 1:length(ms)
        mobius_numbers[ts[i]] = 1 - length(ps[ts[i]])
        for l in 1:(i-1)
            if mat[ts[i], ts[l]] == 1
                mobius_numbers[ts[i]] = mobius_numbers[ts[i]] - mobius_numbers[ts[l]]
            end
        end
    end

    return mobius_numbers
end

function prune_ms_ps(ms, ps, mobius_nos)
    nonzero_mobius = findall(x -> x != 0, mobius_nos)
    return ms[nonzero_mobius], ps[nonzero_mobius], mobius_nos[nonzero_mobius]
end

function children(ms, ps, bs)
    cs = []
    for i in 1:length(bs)
        children = []
        for j in 1:length(ms)
            for k in ps[j]
                if k == i
                    push!(children, j)
                end
            end
        end
        push!(cs, children)
    end
    return cs
end

function calculate_b_nos(ms, ps, mobius_nos)
    return [-(length(ps[i])-1)/mobius_nos[i] for i in 1:length(ms)]
end

function initialize_messages(ms, bs, ps, T)
    ms_dict = Dictionary{Tuple{Int, Int}, ITensor}()
    for (i, m) in enumerate(ms)
        for p in ps[i]
            inds = reduce(vcat, [virtualinds(T, e) for e in filter(x -> x isa NamedEdge, m)])
            if network(T) isa TensorNetworkState
                inds = vcat(inds, prime.(inds))
            end
            msg = ITensor(scalartype(T), 1.0, inds)
            set!(ms_dict, (p, i), msg)
        end       
    end
    return ms_dict
end