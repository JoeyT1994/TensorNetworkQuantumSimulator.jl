using ITensors: ITensors, ITensor, Index, prime, scalar, replaceinds
using Dictionaries: Dictionary

# spinless-fermion single-site matrices (mode basis |0>,|1>), M[out, in]
const _A_MAT    = ComplexF64[0 1; 0 0]   # annihilation:  a|1> = |0>
const _ADAG_MAT = ComplexF64[0 0; 1 0]   # creation:      a†|0> = |1>
const _I2_MAT   = ComplexF64[1 0; 0 1]
const _AUP_MAT = ComplexF64[0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
const _ADN_MAT = ComplexF64[0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0]
const _AUPDAG_MAT = ComplexF64[0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
const _ADNDAG_MAT = ComplexF64[0 0 0 0; 0 0 0 0; 1 0 0 0; 0 -1 0 0]
const _I4_MAT   = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

function spinless(s)
    return dim(s) == 2
end

function fermion_op_matrix(name::String, s::Index)
    if spinless(s) 
        name == "I" && return _I2_MAT
        name == "N" && return _ADAG_MAT * _A_MAT
        (name == "Cdag" || name == "Adag") && return _ADAG_MAT
        (name == "C" || name == "A") && return _A_MAT
    elseif dim(s) == 4
        name == "I" && return _I4_MAT
        name == "Nup" && return _AUPDAG_MAT * _AUP_MAT
        name == "Ndn" && return _ADNDAG_MAT * _ADN_MAT
        name == "NupNdn" && return _AUPDAG_MAT * _AUP_MAT * _ADNDAG_MAT * _ADN_MAT
        (name == "Cupdag" || name == "Aupdag") && return _AUPDAG_MAT
        (name == "Cdndag" || name == "Adndag") && return _ADNDAG_MAT
        (name == "Cup" || name == "Aup") && return _AUP_MAT
        (name == "Cdn" || name == "Adn") && return _ADN_MAT
    end
    return error("Unrecognized fermionic operator string: $name for the given index")
end

is_odd(name::String) = name in ("Cdag", "C", "Adag", "A", "Cupdag", "Cdndag", "Cup", "Cdn", "Aupdag", "Adndag", "Aup", "Adn")

function even_op_tensor(s::Index, name::String, sgr::Vector{Bool})
    M = fermion_op_matrix(name, s)              # M[out, in]
    u = prime(s)
    gr = Dictionary{Index, Vector{Bool}}([u, s], [sgr, sgr])
    return FermionicITensor(ITensor(M, u, s), Index[u, s], Bool[false, true], gr)
end

function odd_op_tensor(s::Index, name::String, d::Index, sgr::Vector{Bool})
    M = fermion_op_matrix(name, s)              # M[out, in]
    u = prime(s)
    gr = Dictionary{Index, Vector{Bool}}([u, s, d], [sgr, sgr, Bool[true]])
    arr = reshape(M, size(M, 1), size(M, 2), 1)
    T = ITensor(arr, u, s, d)
    # Leg ORDER follows Eq.96 exactly (the aux position relative to the physical legs
    # carries a Koszul sign): creation a† = |1⟩⟨0|(1| has the bra aux LAST → [u, s, d];
    # annihilation a = |1)|0⟩⟨1| has the ket aux FIRST → [d, u, s]. (u = out, s = in.)
    if name in ("Cdag", "Adag", "Cupdag", "Aupdag", "Cdndag", "Adndag")               # creation ⇒ bra aux (in) last
        return FermionicITensor(T, Index[u, s, d], Bool[false, true, true], gr)
    else                                      # annihilation ⇒ ket aux (out) first
        return FermionicITensor(T, Index[d, u, s], Bool[false, false, true], gr)
    end
end