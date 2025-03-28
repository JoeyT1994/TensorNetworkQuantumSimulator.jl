function paulirotation(generator, qinds, θ, tninds::IndsNetwork)
    # ("XX", 0.3, (1, 2))) could be the gate part

    # get the right indices from the IndsNetwork
    indices = [only(tninds[ind]) for ind in qinds]

    return paulirotation(generator, θ, indices)
end

"""
    paulirotation(generator::String, θ, indices::Vector{Index})

Returns an ITensor representing a Pauli rotation gate acting on the qubits specified by `indices`. 
The generator is a string of Pauli matrices, e.g. "X" or "Y", "Z", or "XY". 
The angle of rotation is specified by `θ`.
"""
function paulirotation(generator, θ, indices)
    d = getphysicaldim(first(indices))
    heisenberg = d == 4

    nqubits = length(generator)
    @assert length(indices) == nqubits "The number of indices must match the length of the gate generator."
    U = paulirotationmatrix(generator, θ)

    if heisenberg
        # transform into PTMs
        U = PP.calculateptm(U, heisenberg=true)
    end

    # check for physical dimension matching
    # TODO


    # define legs of the tensor
    legs = (indices..., [ind' for ind in indices]...)

    # create the ITensor
    return itensor(transpose(U), legs)

end


"""
    paulirotationmatrix(generator, θ)
"""
function paulirotationmatrix(generator, θ)
    symbols = stringtosymbols(generator)
    pauli_rot = PP.PauliRotation(symbols, 1:length(symbols))
    return PP.tomatrix(pauli_rot, θ)
end