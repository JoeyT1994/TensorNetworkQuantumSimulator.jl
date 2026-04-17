using ITensors: hastags

# conversion of a tuple circuit to an ITensor circuit
function toitensor(circuit::Vector, g::NamedGraph, sinds::Dictionary)
    return [toitensor(gate, g, sinds) for gate in circuit]
end

#Determine if a string represents a pauli string
function _ispaulistring(string::String)
    return all(s ∈ ['X', 'Y', 'Z', 'x', 'y', 'z'] for s in string)
end

#Gates which take a single theta argument (rotation argument)
function _takes_theta_argument(string::String)
    return string ∈ ["Rx", "Ry", "Rz", "CRx", "CRy", "CRz", "Rxxyy", "Rxxyyzz", "Rz+", "Rz+z+"]
end

#Gates which take a single phi argument (rotation argument)
function _takes_phi_argument(string::String)
    return string ∈ ["Rxx", "Ryy", "Rzz", "P", "CPHASE"]
end


#Gates which need to have parameter rescaled to match qiskit convention
function param_rescaling(string::String, param::Number)
    string ∈ ["Rxx", "Ryy", "Rzz"] && return param / 2
    return param
end

#Convert a gate to the corresponding ITensor
function toitensor(gate::Tuple, g::NamedGraph, siteinds::Dictionary)
    gate_symbol = gate[1]
    gate_verts = collect_vertices(gate[2], g)
    s_inds = [only(siteinds[v]) for v in gate_verts]

    all(map(sind -> hastags(sind, "Pauli"), s_inds)) &&
        return toitensor_heisenberg(gate_symbol, gate[3], s_inds), gate_verts

    if _ispaulistring(gate_symbol)
        gate =
            prod(ITensors.op(string(op), sind) for (op, sind) in zip(gate_symbol, s_inds))
    elseif length(gate) == 2
        gate = ITensors.op(gate_symbol, s_inds...)
    elseif _takes_theta_argument(gate_symbol)
        gate = ITensors.op(gate_symbol, s_inds...; θ = param_rescaling(gate_symbol, gate[3]))
    elseif _takes_phi_argument(gate_symbol)
        gate = ITensors.op(gate_symbol, s_inds...; ϕ = param_rescaling(gate_symbol, gate[3]))
    else
        throw(ArgumentError("Wrong gate format"))
    end
    return gate, gate_verts
end


"""
    paulirotationmatrix(generator, θ)
"""
function paulirotationmatrix(generator, θ)
    symbols = [Symbol(s) for s in generator]
    pauli_rot = PP.PauliRotation(symbols, 1:length(symbols))
    return PP.tomatrix(pauli_rot, θ)
end

#Convert a gate that's in the Heisenberg picture to an ITensor for the Pauli Transfer Matrix
function toitensor_heisenberg(generator, θ, indices)
    @assert first(generator) == 'R'
    generator = generator[2:length(generator)]
    @assert _ispaulistring(generator)
    generator = uppercase.(generator)
    U = paulirotationmatrix(generator, θ)
    U = PP.calculateptm(U, heisenberg = true)

    # check for physical dimension matching
    # TODO

    # define legs of the tensor
    legs = (indices..., [ind' for ind in indices]...)

    # create the ITensor
    return itensor(transpose(U), legs)
end

#Return itself as the type is already correct
function toitensor(gate::ITensor, sinds::Dictionary)
    return gate
end

"""
    ITensors.op(::OpName"Rxxyy", ::SiteType"S=1/2"; θ::Number)

Gate for rotation by XXYY at a given angle
"""
function ITensors.op(::OpName"Rxxyy", ::SiteType"S=1/2", s1::Index, s2::Index; θ =1)
    h = 0.5*(op("X", s1) * op("X", s2) + op("Y",s1) * op("Y", s2))
    return exp( -im * θ * h)
end

"""
    ITensors.op(::OpName"Rxxyyzz", ::SiteType"S=1/2"; θ::Number)

Gate for rotation by XXYYZZ at a given angle
"""
function ITensors.op(::OpName"Rxxyyzz", ::SiteType"S=1/2", s1::Index, s2::Index; θ =1)
    h = 0.5*(op("X", s1) * op("X", s2) + op("Y",s1) * op("Y", s2) + op("Z", s1) * op("Z", s2))
    return exp( -im * θ * h)
end

ITensors.op(o::OpName"Rxxyy", ::SiteType"Qubit"; θ::Number) = ITensors.op(o, ITensorMPS.SiteType("S=1/2"); θ)
ITensors.op(o::OpName"Rxxyyzz", ::SiteType"Qubit"; θ::Number) = ITensors.op(o, ITensorMPS.SiteType("S=1/2"); θ)
