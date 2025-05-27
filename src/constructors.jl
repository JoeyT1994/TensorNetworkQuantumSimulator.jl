const stringtointmap = Dict("I" => 1, "X" => 2, "Y" => 3, "Z" => 4)

"""
    zerostate(g::NamedGraph; pauli_basis = false)

Tensor network for vacuum state on given graph, i.e all spins up
"""
function zerostate(g::NamedGraph; pauli_basis = false)
    if !pauli_basis
        # the most common case 
        return zerostate(siteinds("Qubit", g))
    else
        return zerostate(siteinds(4, g))
    end
end

"""
    zerostate(indices::IndsNetwork)

Tensor network for vacuum state on given indsnetwork
"""
function zerostate(indices::IndsNetwork)
    inds = reduce(vcat, [indices[v] for v in vertices(indices)])
    dims = dim.(inds)
    if all(d -> d== 2, dims)
        return ITensorNetwork(v -> [1.0, 0.0], indices)
    elseif all(d -> d== 4, dims)
        return ITensorNetwork(v -> [1.0, 0.0, 0.0, 1.0], indices)
    else
        throw(ArgumentError("Only physical dimensions 2 and 4 are supported."))
    end
end

"""
    topaulitensornetwork(op, g::NamedGraph)

Tensor network (in Heisenberg picture) for given pauli string on given graph
"""
function topaulitensornetwork(op, g::NamedGraph)
    return topaulitensornetwork(op, siteinds(4, g))
end

""" 
    topaulitensornetwork(op, tninds::IndsNetwork)

Tensor network (in Heisenberg picture) for given pauli string on given IndsNetwork
"""
function topaulitensornetwork(op, tninds::IndsNetwork)
    nq = getnqubits(tninds)

    op_string = op[1] # could be "XX", "Y", "Z"
    op_inds = op[2]  # could be [1, 2], [1], [2]
    if length(op) == 2
        op_coeff = 1.0
    elseif length(op) == 3
        op_coeff = op[3]
    else
        throw(ArgumentError("Wrong Operator format"))
    end

    # verify that the operator is acting on the correct number of qubits
    @assert length(op_inds) == length(op_string) "Pauli string $(op_string) does not match the number of indices $(op_inds)."

    # verify that all op_inds are actually in tninds 
    all_inds = vertices(tninds)
    for ind in op_inds
        if !(ind in all_inds)
            throw(
                ArgumentError(
                    "Index $ind of the operator is not in the IndsNetwork $tninds.",
                ),
            )
        end
    end

    function map_f(ind)
        pos = findfirst(i -> i == ind, op_inds)
        if isnothing(pos)
            #the identity case
            return [1.0, 0.0, 0.0, 0.0]
        end

        # only give the first element the op coefficient
        if pos == 1
            coeff = op_coeff
        else
            coeff = 1.0
        end

        # pos should now be "I", "X", "Y", or "Z"
        pauli = string(op_string[pos])
        one_pos = stringtointmap[pauli]
        vec = zeros(4)
        coeff
        vec[one_pos] = coeff
        return vec
    end


    return ITensorNetwork(map_f, tninds)
end

"""
    identitytensornetwork(tninds::IndsNetwork)

Tensor network (in Heisenberg picture) for identity matrix on given IndsNetwork
"""
function identitytensornetwork(tninds::IndsNetwork)
    return topaulitensornetwork(("I", [first(vertices(tninds))]), tninds)
end