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
        name == "Sz" && return 0.5*(_AUPDAG_MAT * _AUP_MAT - _ADNDAG_MAT * _ADN_MAT)
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

# Local product-state vector for a named fermionic basis state. Spinless (dim 2):
# |0⟩/|1⟩; spinful (dim 4): |0⟩, |↑⟩, |↓⟩, |↑↓⟩.
function fermionic_statevector(name::String, sind::Index)
    d = dim(sind)
    if d == 2
        name in ("0", "Emp", "Empty") && return ComplexF64[1, 0]
        name in ("1", "Occ", "Occupied") && return ComplexF64[0, 1]
        error("Unrecognized spinless fermion state \"$name\". Supported: 0/Emp, 1/Occ.")
    elseif d == 4
        name in ("0", "Emp", "Empty") && return ComplexF64[1, 0, 0, 0]
        name in ("Up", "↑") && return ComplexF64[0, 1, 0, 0]
        name in ("Dn", "Down", "↓") && return ComplexF64[0, 0, 1, 0]
        name in ("UpDn", "↑↓", "2") && return ComplexF64[0, 0, 0, 1]
        error("Unrecognized spinful fermion state \"$name\". Supported: 0/Emp, Up/↑, Dn/↓, UpDn/↑↓.")
    end
    error("Fermionic product states support dimension-2 or dimension-4 sites only.")
end

# Parity (false = even, true = odd) of a parity-DEFINITE local state. Errors if the
# vector has weight in both sectors — coherent parity superpositions are forbidden by
# fermion-parity superselection and would make the local tensor parity-indefinite.
function fermionic_state_parity(vec::AbstractVector, gr::Vector{Bool})
    nz = findall(!iszero, vec)
    isempty(nz) && error("Local fermion state is the zero vector.")
    ps = unique(gr[i] for i in nz)
    length(ps) == 1 || error("Local fermion state mixes even and odd parity sectors; fermion-parity superselection forbids this.")
    return only(ps)
end