using TensorNetworkQuantumSimulator
using NamedGraphs: unique_simplecycles_limited_length
using ITensors: Index

using Graphs: topological_sort
using Graphs.SimpleGraphs: SimpleDiGraph

function special_multiply(t1::ITensor, t2::ITensor)
    cinds = commoninds(t1, t2)
    ds = []
    for cind in cinds
        t1 = replaceind(t1, cind, cind')
        t2 = replaceind(t2, cind, cind'')
        push!(ds, delta([cind, cind', cind'']))
    end

    t = reduce(*, [[t1, t2 ]; ds])
    return t    
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


function construct_bp_bs(t::AbstractTensorNetwork)
    return collect([[i for i in inds(t[v])] for v in vertices(t)])
end

function construct_gbp_bs(t::AbstractTensorNetwork)
    bs = construct_bp_bs(t)
    cycles = unique_simplecycles_limited_length(t, 4)
    gbp_bs = copy(bs)
    for cycle in cycles
        is = Index[]
        for (i, v) in enumerate(cycle)
            if i != length(cycle)
                index = only(virtualinds(t, NamedEdge(v => cycle[i+1])))
            else
                index = only(virtualinds(t, NamedEdge(v => cycle[1])))
            end
            push!(is, index)
        end
        push!(gbp_bs, is)  # Add first vertex to close the loop
    end

    return gbp_bs
end

function intersections(ms)
    intersects = []
    for i in 1:length(ms), j in i+1:length(ms)
        s = intersect(Set(ms[i]), Set(ms[j]))
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
            if Set(m) ∉ all_ms
                push!(all_ms, Set(m))
            end
        end
        current_ms = intersections(current_ms)
    end

    return collect.(all_ms)
end

function parents(m, bs)
    parents = []
    for (i, b) in enumerate(bs)
        if issubset(Set(m), Set(b))
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
        if issubset(Set(m1), Set(m2))
            mat[i, j + i] = 1
        end
        if i != j && issubset(Set(m2), Set(m1))
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

function get_psis(bs, T::TensorNetwork; include_factors = true)
    potentials = []
    for b in bs
        pot = ITensor(scalartype(T), 1.0, b)
        for v in vertices(T)
            inds_v = inds(T[v])
            if issubset(Set(inds_v), Set(b)) && include_factors
                pot = special_multiply(pot, T[v])
            end
        end
        push!(potentials, pot)
    end
    return potentials
end

function initialize_messages(ms, bs, ps, T)
    ms_dict = Dictionary{Tuple{Int, Int}, ITensor}()
    for (i, m) in enumerate(ms)
        for p in ps[i]
            set!(ms_dict, (p, i), ITensor(scalartype(T), 1.0, m))
        end       
    end
    return ms_dict
end

function initialize_beliefs(psis)
    beliefs = []
    for psi in psis
        z = real(sum(psi))
        push!(beliefs, psi / z)
    end
    return beliefs
end