using ITensors: ITensors, ITensor, Index, prime, scalar, replaceinds
using Dictionaries: Dictionary

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

is_odd(name::String) = name in ("Cdag", "C", "Adag", "A")

function even_op_tensor(s::Index, name::String, sgr::Vector{Bool})
    M = _fermion_op_matrix(name)              # M[out, in]
    u = prime(s)
    gr = Dictionary{Index, Vector{Bool}}([u, s], [sgr, sgr])
    return FermionicITensor(ITensor(M, u, s), Index[u, s], Bool[false, true], gr)
end

function odd_op_tensor(s::Index, name::String, d::Index, sgr::Vector{Bool})
    M = _fermion_op_matrix(name)              # M[out, in]
    u = prime(s)
    gr = Dictionary{Index, Vector{Bool}}([u, s, d], [sgr, sgr, Bool[true]])
    arr = reshape(M, size(M, 1), size(M, 2), 1)
    T = ITensor(arr, u, s, d)
    # Leg ORDER follows Eq.96 exactly (the aux position relative to the physical legs
    # carries a Koszul sign): creation a† = |1⟩⟨0|(1| has the bra aux LAST → [u, s, d];
    # annihilation a = |1)|0⟩⟨1| has the ket aux FIRST → [d, u, s]. (u = out, s = in.)
    if name in ("Cdag", "Adag")               # creation ⇒ bra aux (in) last
        return FermionicITensor(T, Index[u, s, d], Bool[false, true, true], gr)
    else                                      # annihilation ⇒ ket aux (out) first
        return FermionicITensor(T, Index[d, u, s], Bool[false, false, true], gr)
    end
end