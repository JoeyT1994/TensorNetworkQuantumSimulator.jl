getnqubits(g::NamedGraph) = length(g.vertices)
getnqubits(tninds::IndsNetwork) = length(tninds.data_graph.vertex_data)


getphysicaldim(indices::IndsNetwork) = dim(first(indices.data_graph.vertex_data.values))
getphysicaldim(indices::AbstractVector{<:Index}) = first(indices)
getphysicaldim(index::Index) = dim(index)


## Utilities to globally set bp_update_kwargs
const _default_bp_update_maxiter = 20
const _default_bp_update_tol = 1e-8
_default_message_update_function(ms) = make_eigs_real.(default_message_update(ms))

const _default_bp_update_kwargs = (
    maxiter=_default_bp_update_maxiter,
    tol=_default_bp_update_tol,
    message_update=_default_message_update_function
)

# we make this a Dict that it can be pushed to with kwargs that we haven't thought of
const _global_bp_update_kwargs::Dict{Symbol,Any} = Dict(
    :maxiter => _default_bp_update_maxiter,
    :tol => _default_bp_update_tol,
    :message_update_kwargs => (; message_update_function=_default_message_update_function)
)

function set_global_bp_update_kwargs!(; kwargs...)
    for (arg, val) in kwargs
        _global_bp_update_kwargs[arg] = val
    end
    return get_global_bp_update_kwargs()
end

function get_global_bp_update_kwargs()
    # return as a named tuple
    return (; _global_bp_update_kwargs...)
end

function reset_global_bp_update_kwargs!()
    empty!(_global_bp_update_kwargs)
    _global_bp_update_kwargs[:maxiter] = _default_bp_update_maxiter
    _global_bp_update_kwargs[:tol] = _default_bp_update_tol
    _global_bp_update_kwargs[:message_update_kwargs] = _default_message_update_function
    return get_global_bp_update_kwargs()
end


## 
function getmode(indices::IndsNetwork)
    d = getphysicaldim(indices)
    if d == 2
        return "Schrödinger"
    elseif d == 4
        return "Heisenberg"
    else
        throwdimensionerror()
    end
end

function trace(Q::ITensorNetwork)
    d = getphysicaldim(siteinds(Q))
    if d == 2
        vec = [1.0, 1.0]
    elseif d == 4
        vec = [1.0, 0.0, 0.0, 0.0]
    else
        throwdimensionerror()
    end

    val = ITensorNetworks.inner(ITensorNetwork(v -> vec, siteinds(Q)), Q; alg="bp")
    return val
end


function topologytograph(topology)
    # TODO: adapt this to named graphs with non-integer labels
    # find number of vertices
    nq = maximum(maximum.(topology))
    adjm = zeros(Int, nq, nq)
    for (ii, jj) in topology
        adjm[ii, jj] = adjm[jj, ii] = 1
    end
    return NamedGraph(SimpleGraph(adjm))
end


function graphtotopology(g)
    return [[edge.src, edge.dst] for edge in edges(g)]
end

stringtosymbols(str) = [Symbol(s) for s in str]

function throwdimensionerror()
    throw(ArgumentError("Only physical dimensions 2 and 4 are supported."))
end

# 