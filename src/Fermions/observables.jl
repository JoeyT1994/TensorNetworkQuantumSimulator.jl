# Observables for FermionicTensorNetworkState (exact, dense).
#
# ⟨ψ|O|ψ⟩ is computed as ⟨dag(K) , O·K⟩ where
#   * K   = the fully contracted ket (open legs = the site indices), and
#   * O·K = K with the single-site operator factors of O applied to its site legs.
# The bra is `fermionic_dag(K)` and is glued to `O·K` directly over the site legs
# (one contraction per site). This uses only the validated `fermionic_contract` /
# `fermionic_dag` primitives: ⟨dag(K), K⟩ reproduces Σ|amp|² exactly.
#
# IMPORTANT: do NOT split each site bond by inserting a separate primed identity
# "wire" tensor (ket leg s → op → bra leg s'). That doubling drops the bond-parity
# factor `g` that a single fermionic site contraction carries, giving a wrong norm.
#
# A single-site operator that changes fermion parity (Cdag/C) is ODD. Following
# arXiv:2410.02215 §II E, each odd operator carries an extra dummy odd index so the
# operator tensor is parity even; the dummy of one odd operator contracts with the
# dummy of its partner (the operator "string"). We support observables with 0 or 2
# odd operators (identity, number, density-density, single hopping c_i†c_j). An odd
# number of odd operators has zero expectation by parity.

using ITensors: ITensors, ITensor, Index, prime, scalar, replaceind
using Dictionaries: Dictionary, set!

_is_odd_op(name::String) = name in ("Cdag", "C", "Adag", "A")

# Pull the per-vertex ket tensors out as FermionicTensors (open legs = sites + bonds).
function _ket_fermionic_tensors(ψ::FermionicTensorNetworkState, gr::Dictionary)
    return [
        FermionicTensor(ψ[v], copy(index_order(ψ)[v]), copy(index_directions(ψ)[v]), gr)
        for v in vertices(ψ)
    ]
end

# Walk the nested binary contraction tree `seq` (integer indices into `fts`),
# folding pairs together with `fermionic_contract`. Because fermionic_contract is
# contraction-order independent for parity-even tensors, the result equals the
# naive fold; the ordering only changes intermediate cost.
function _follow_sequence(seq, fts::Vector{<:FermionicTensor})
    seq isa Integer && return fts[seq]
    acc = _follow_sequence(seq[1], fts)
    for k in 2:length(seq)
        acc = fermionic_contract(acc, _follow_sequence(seq[k], fts))
    end
    return acc
end

# Contract a list of FermionicTensors using the bosonic optimal contraction order
# (`contraction_sequence(..., alg="optimal")`) on the underlying ITensors, then
# follow that tree through fermionic_contract.
function _contract_fermionic_tensors(fts::Vector{<:FermionicTensor}; alg = "optimal")
    length(fts) == 1 && return only(fts)
    seq = contraction_sequence([ft.tensor for ft in fts]; alg)
    return _follow_sequence(seq, fts)
end

# Fully contract the ket into a single FermionicTensor whose open legs are the sites.
# Naive vertex-order fold — kept as a simple reference for testing. Production paths
# (norm_sqr / expect) use the optimal-order contractor instead.
function _contract_ket(ψ::FermionicTensorNetworkState, gr::Dictionary)
    fts = _ket_fermionic_tensors(ψ, gr)
    K = fts[1]
    for k in 2:length(fts)
        K = fermionic_contract(K, fts[k])
    end
    return K
end

# Apply a single-site operator `M = _fermion_op_matrix(name)` to leg `s` of `OK`.
# The op tensor has `s` incoming (arrow OK→op, matching the ket's outgoing site leg)
# and a fresh outgoing leg carrying the result; that leg is relabelled back to `s`
# so the operator maps s → s in place. `dummy` (if given) is an extra dim-1 odd leg
# (the operator-string bond) with arrow direction `dummy_dir`.
function _apply_site_op(OK::FermionicTensor, s::Index, name::String, gr::Dictionary; dummy = nothing, dummy_dir::Bool = false)
    M = _fermion_op_matrix(name)              # M[out, in]
    u = prime(s)
    haskey(gr, u) || set!(gr, u, gr[s])
    # Declare the operator in natural ket-bra order [u (out), s (in)] so that contracting
    # `s` (with the operator passed first) needs NO internal reorder — a [s, u] order would
    # force fc to swap the two odd legs, spuriously negating parity-odd components.
    op = if dummy === nothing
        FermionicTensor(ITensor(M, u, s), Index[u, s], Bool[false, true], gr)
    else
        arr = reshape(M, size(M, 1), size(M, 2), 1)
        FermionicTensor(ITensor(arr, u, s, dummy), Index[u, s, dummy], Bool[false, true, dummy_dir], gr)
    end
    # Operator application is a BRA-KET contraction (Eq.24, the plain contraction map
    # C: V*⊗V → ℂ, NO supertrace): the operator's site leg `s` is a bra (in) and the
    # ket's site leg is a ket (out). Pass the operator FIRST so that in fermionic_contract
    # A=op holds `s` as in ⇒ no bond-parity g is inserted (a ket-first ordering would add a
    # spurious supertrace and flip the sign on the occupied component).
    R = fermionic_contract(op, OK)            # consumes s, opens u (= prime(s)) (+ dummy)
    T = replaceind(R.tensor, u, s)
    neworder = Index[i == u ? s : i for i in R.order]
    return FermionicTensor(T, neworder, R.dirs, gr)
end

# Build a single-site ODD operator tensor with a shared auxiliary odd leg `d`.
# Per SciPost arXiv "Fermionic tensor networks" Eq.96, the creation operator carries
# its auxiliary leg as a BRA, a† = |1⟩⟨0|(1| (incoming, dir=true), and the annihilation
# operator carries it as a KET, a = |1)|0⟩⟨1| (outgoing, dir=false). Each such tensor is
# parity EVEN (the odd physical change is balanced by the odd aux leg). Legs: [s (in),
# u=s' (out), d]. Contracting two of these over `d` reproduces the (even) two-site term
# (Eq.95), which we then apply to the ket as an ordinary even operator.
function _odd_op_tensor(s::Index, name::String, d::Index, gr::Dictionary)
    M = _fermion_op_matrix(name)              # M[out, in]
    u = prime(s)
    haskey(gr, u) || set!(gr, u, gr[s])
    arr = reshape(M, size(M, 1), size(M, 2), 1)
    T = ITensor(arr, u, s, d)
    # Leg ORDER follows Eq.96 exactly (the aux position relative to the physical legs
    # carries a Koszul sign): creation a† = |1⟩⟨0|(1| has the bra aux LAST → [u, s, d];
    # annihilation a = |1)|0⟩⟨1| has the ket aux FIRST → [d, u, s]. (u = out, s = in.)
    if name in ("Cdag", "Adag")               # creation ⇒ bra aux (in) last
        return FermionicTensor(T, Index[u, s, d], Bool[false, true, true], gr)
    else                                      # annihilation ⇒ ket aux (out) first
        return FermionicTensor(T, Index[d, u, s], Bool[false, false, true], gr)
    end
end

# Build O·K: apply every non-identity single-site operator factor of `O` to the ket.
# Returns `nothing` when the operator has odd total parity (⟨O⟩ = 0 by parity).
#
# Even single-site factors are applied one at a time (single-bond site contraction).
# A pair of ODD factors (e.g. a hopping c_i† c_j) is handled by FIRST contracting the
# two odd single-site operator tensors through their shared auxiliary leg into the even
# two-site operator (Eq.95-97), then applying that even operator to the ket. Building the
# two-site term before applying it keeps the operator-string bond a clean single-bond
# contraction, rather than entangling it with a site bond in one multi-bond contraction.
function _apply_operator(ψ::FermionicTensorNetworkState, K::FermionicTensor, op_string_f::Function, gr::Dictionary)
    # collect odd operators (each contributes one dummy odd leg)
    odd_sites = Tuple{Index, String}[]
    for v in vertices(ψ)
        name = op_string_f(v)
        if _is_odd_op(name)
            for s in siteinds(ψ, v)
                push!(odd_sites, (s, name))
            end
        end
    end
    isodd(length(odd_sites)) && return nothing          # parity-forbidden ⇒ ⟨O⟩ = 0
    length(odd_sites) > 2 && error("Fermionic expect currently supports at most one pair of odd operators (e.g. a single hopping term).")

    # Rebind K onto `gr` so every tensor below shares ONE grading dictionary (the
    # primed `u` and dummy `d` indices get registered here, and fc looks them up via
    # `OK.grading`). `gr` already contains all of K's open (site) indices.
    OK = FermionicTensor(K.tensor, copy(K.order), copy(K.dirs), gr)
    # apply all EVEN single-site factors first (each a single-bond site contraction)
    for v in vertices(ψ)
        name = op_string_f(v)
        (name == "I" || name == "ρ") && continue
        _is_odd_op(name) && continue
        for s in siteinds(ψ, v)
            dim(s) == 2 || error("Fermionic measurement currently supports spinless (dimension-2) sites only.")
            OK = _apply_site_op(OK, s, name, gr)
        end
    end

    # apply the odd pair as a pre-contracted even two-site operator
    if length(odd_sites) == 2
        (s1, n1), (s2, n2) = odd_sites
        dim(s1) == 2 && dim(s2) == 2 || error("Fermionic measurement currently supports spinless (dimension-2) sites only.")
        d = Index(1, "Fermion,OpString")
        set!(gr, d, Bool[true])                         # odd auxiliary (operator-string) bond
        # Contract the operator-string bond ANNIHILATION-first. The annihilation operator
        # carries its aux as a KET (out), the creation operator as a BRA (in); per Eq.107 a
        # ket-bra contraction carries a supertrace, which fc inserts iff the operand holding
        # the bond as a ket (out) is passed first. This supplies the (−1) that matches the
        # fermionic ordering of the operator product c_i† c_j.
        opA = _odd_op_tensor(s1, n1, d, gr)
        opB = _odd_op_tensor(s2, n2, d, gr)
        ann_first = n1 in ("C", "A")                    # annihilation (ket aux) operand first
        O2 = ann_first ? fermionic_contract(opA, opB) : fermionic_contract(opB, opA)
        OK = fermionic_contract(O2, OK)                 # op-first (bra-ket): standard, no supertrace
        u1, u2 = prime(s1), prime(s2)
        T = replaceind(replaceind(OK.tensor, u1, s1), u2, s2)
        neworder = Index[i == u1 ? s1 : (i == u2 ? s2 : i) for i in OK.order]
        OK = FermionicTensor(T, neworder, OK.dirs, gr)
    end
    return OK
end

# ⟨ψ|O|ψ⟩ (unnormalised): glue dag(K) to O·K over the site legs.
function _expectation_numerator(ψ::FermionicTensorNetworkState, K::FermionicTensor, op_string_f::Function, gr::Dictionary)
    OK = _apply_operator(ψ, K, op_string_f, gr)
    OK === nothing && return zero(ComplexF64)
    return scalar(fermionic_contract(fermionic_dag(K), OK).tensor)
end

# ---- norm_sqr ----
function norm_sqr(alg::Algorithm"exact", ψ::FermionicTensorNetworkState)
    gr = copy(grading(ψ))
    K = _contract_fermionic_tensors(_ket_fermionic_tensors(ψ, gr))
    return scalar(fermionic_contract(fermionic_dag(K), K).tensor)
end

function norm_sqr(ψ::FermionicTensorNetworkState; alg = "exact", kwargs...)
    alg == "exact" || error("Only alg=\"exact\" is currently supported for FermionicTensorNetworkState.")
    return norm_sqr(Algorithm(alg), ψ; kwargs...)
end

# ---- expect ----
function expect(alg::Algorithm"exact", ψ::FermionicTensorNetworkState, observables::Vector{<:Tuple})
    ITensors.disable_warn_order()
    gr = copy(grading(ψ))
    K = _contract_fermionic_tensors(_ket_fermionic_tensors(ψ, gr))
    denom = scalar(fermionic_contract(fermionic_dag(K), K).tensor)
    out = Number[]
    for obs in observables
        op_strings, vs, coeff = collectobservable(obs, graph(ψ))
        if iszero(coeff)
            push!(out, zero(coeff))
            continue
        end
        op_dict = Dict(zip(vs, op_strings))
        op_string_f = v -> get(op_dict, v, "I")
        numer = _expectation_numerator(ψ, K, op_string_f, copy(gr))
        push!(out, coeff * (numer / denom))
    end
    return out
end

expect(alg::Algorithm"exact", ψ::FermionicTensorNetworkState, observable::Tuple) =
    only(expect(alg, ψ, [observable]))

function expect(ψ::FermionicTensorNetworkState, observable; alg::Union{String, Nothing} = "exact", kwargs...)
    alg == "exact" || error("Only alg=\"exact\" is currently supported for FermionicTensorNetworkState.")
    return expect(Algorithm(alg), ψ, observable; kwargs...)
end
