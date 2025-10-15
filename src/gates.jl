# conversion of a tuple circuit to an ITensor circuit
function toitensor(circuit, sinds::IndsNetwork)
    return [toitensor(gate, sinds) for gate in circuit]
end

#Determine if a string represents a pauli string
function _ispaulistring(string::String)
    return all(s ∈ ['X', 'Y', 'X', 'x', 'y', 'z'] for s in string)
end

#Gates which take a single theta argument (rotation argument)
function _takes_theta_argument(string::String)
    return string ∈ ["Rx", "Ry", "Rz", "CRx", "CRy", "CRz", "Rxxyy", "Rxxyyzz"]
end

#Gates which take a single phi argument (rotation argument)
function _takes_phi_argument(string::String)
    return string ∈ ["Rxx", "Ryy", "Rzz", "P", "CPHASE"]
end

function _takes_theta_beta_argument(string::String)
    return string ∈ ["xx_plus_yy"]
end

#Gates which need to have parameter rescaled to match qiskit convention
function param_rescaling(string::String, param::Number)
    string ∈ ["Rxx", "Ryy", "Rzz", "Rxxyy", "Rxxyyzz"] && return param / 2
    return param
end

#Convert a gate to the corrresponding ITensor
function toitensor(gate::Tuple, sinds::IndsNetwork)

    gate_symbol = gate[1]
    gate_inds = gate[2]
    # if it is a NamedEdge, we need to convert it to a tuple
    gate_inds = _ensuretuple(gate_inds)
    s_inds = [only(sinds[v]) for v in gate_inds]

    all(map(sind -> dim(sind) == 4, s_inds)) &&
        return toitensor_heisenberg(gate_symbol, gate[3], s_inds)

    if _ispaulistring(gate_symbol)
        gate =
            prod(ITensors.op(string(op), sind) for (op, sind) in zip(gate_symbol, s_inds))
    elseif length(gate) == 2
        gate = ITensors.op(gate_symbol, s_inds...)
    elseif _takes_theta_argument(gate_symbol)
        gate = ITensors.op(gate_symbol, s_inds...; θ = param_rescaling(gate_symbol, gate[3]))
    elseif _takes_phi_argument(gate_symbol)
        gate = ITensors.op(gate_symbol, s_inds...; ϕ = param_rescaling(gate_symbol, gate[3]))
    elseif _takes_theta_beta_argument(gate_symbol)
        gate = ITensors.op(gate_symbol, s_inds...; θ = first(gate[3]), β = last(gate[3]))
    else
        throw(ArgumentError("Wrong gate format"))
    end

    return gate
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
function toitensor(gate::ITensor, sinds::IndsNetwork)
    return gate
end

#Conversion of the gate indices to a tuple
function _ensuretuple(gate_inds::Union{Tuple,AbstractArray})
    return gate_inds
end

#Conversion of a NamedEdge to a tuple
function _ensuretuple(gate_inds::NamedEdge)
    return (gate_inds.src, gate_inds.dst)
end

"""
    ITensors.op(::OpName"xx_plus_yy", ::SiteType"S=1/2"; θ::Number, β::Number)

Gate for rotation by XX+YY at a given angle with Rz rotations either size. Consistent with qiskit.
"""
function ITensors.op(::OpName"xx_plus_yy", ::SiteType"S=1/2"; θ::Number, β::Number)
    return [
        [1 0 0 0];
        [0 cos(θ / 2) -im * sin(θ / 2) * exp(-im * β) 0]
        [0 -im * sin(θ / 2) * exp(im * β) cos(θ / 2) 0]
        [0 0 0 1]
    ]
end
 
"""
    ITensors.op(::OpName"Rxxyy", ::SiteType"S=1/2"; θ::Number)

Gate for rotation by XXYY at a given angle
"""
function ITensors.op(
    ::OpName"Rxxyy", ::SiteType"S=1/2"; θ::Number
  )
    mat = zeros(ComplexF64, 4, 4)
    mat[1, 1] = 1
    mat[4, 4] = 1
    mat[2, 2] = cos(θ)
    mat[2, 3] = -1.0 * im * sin(θ)
    mat[3, 2] = -1.0 * im * sin(θ)
    mat[3, 3] = cos(θ)
    return mat
end

"""
    ITensors.op(::OpName"Rxxyyzz", ::SiteType"S=1/2"; θ::Number)

Gate for rotation by XXYYZZ at a given angle
"""
function ITensors.op(
    ::OpName"Rxxyyzz", ::SiteType"S=1/2"; θ::Number
  )
    a = exp( im * θ * 0.5)
    mat = zeros(ComplexF64, 4, 4)
    mat[1, 1] = conj(a)
    mat[2, 2] = cos(θ) * a
    mat[2, 3] = -1.0 * im * a * sin(θ)
    mat[3, 2] = -1.0 * im * a * sin(θ)
    mat[3, 3] = cos(θ) * a
    mat[4,4] = conj(a)
    return mat
end

ITensors.op(o::OpName"Rxxyy", ::SiteType"Qubit"; θ::Number) = ITensors.op(o, ITensorMPS.SiteType("S=1/2"); θ)
ITensors.op(o::OpName"Rxxyyzz", ::SiteType"Qubit"; θ::Number) = ITensors.op(o, ITensorMPS.SiteType("S=1/2"); θ)