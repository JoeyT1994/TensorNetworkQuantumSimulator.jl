# spinless-fermion single-site matrices (mode basis |0>,|1>), M[out, in]
const _A_MAT    = ComplexF64[0 1; 0 0]   # annihilation:  a|1> = |0>
const _ADAG_MAT = ComplexF64[0 0; 1 0]   # creation:      a†|0> = |1>
const _I2_MAT   = ComplexF64[1 0; 0 1]

function _fermion_op_matrix(name::String)
    name == "I" && return _I2_MAT
    name == "N" && return _ADAG_MAT * _A_MAT
    (name == "Cdag" || name == "Adag") && return _ADAG_MAT
    (name == "C" || name == "A") && return _A_MAT
    return error("Unrecognized fermionic operator string: $name")
end