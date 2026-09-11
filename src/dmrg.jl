#=
Variational ground states from belief propagation via the generating-function construction.

The Hamiltonian is never contracted as the indefinite network ⟨ψ|H|ψ⟩. Instead it is embedded in
the POSITIVE norm network of the generating operator, T(λ) = ⟨ψ|G(λ)|ψ⟩ with G(λ) = 1 + λH + O(λ²)
(see `GeneratingOperator`), so that every environment is a positive double-layer environment and
the whole BP/gauge machinery of the package applies unchanged. With X(λ) the BP fixed point of
T(λ) and Z_B the Bethe estimator:

  energy         E_B = ∂λ Z_B(T(λ), X₀)|₀ / Z_B(T₀, X₀)        (messages held fixed: envelope theorem)
  response       X′ = dX/dλ|₀ from the linearised message-passing fixed-point equation
  local problem  H_eff ψᵥ = E N_eff ψᵥ, with N_eff the norm environment of vertex v and
                 H_eff ψᵥ = ∂λ ∂ψᵥ* Z_B = [explicit insertion at v] + [environment response X′]

At χ = 1 (this file) the linear response is the one-insertion sector of the messages; on a tree it
is exact and the local problem is one-site DMRG. The same construction carries over to the
matrix-product / CTM environments (MP-BP), where it is what makes energies O(ε²).
=#

using KrylovKit: eigsolve

function _region_scalar(ts::Vector)
    sequence = contraction_sequence(ts; alg = "optimal")
    return scalar(contract(ts; sequence))
end

"""
    generating_cache(ψ::TensorNetworkState, gen::GeneratingOperator; kwargs...)

Converged belief-propagation cache of the generating norm network ⟨ψ|G(0)|ψ⟩ (a `QuadraticForm`
whose operator layer is `gen.value`). Its messages carry the ket bond, the auxiliary index and the
bra bond; their `a = 0` components are the norm messages and the `a > 0` components the
half-insertions `ψ†Aₐψ` propagated from the left ends of the Hamiltonian bonds. `kwargs` go to
`update`.
"""
function generating_cache(ψ::TensorNetworkState, gen::GeneratingOperator; kwargs...)
    return _hermitian_gauge!(update(BeliefPropagationCache(QuadraticForm(ψ, gen.value)); kwargs...))
end

# The `a = 0` (norm-sector) slice of a message or message-like tensor, and its entry sum. The sum
# of a Hermitian matrix's entries is real, so it is a Hermiticity-preserving normalisation.
function _value_slice(t)
    aux = filter(i -> occursin("aux", string(TensorInterface.tags(i))), collect(TensorInterface.inds(t)))
    isempty(aux) && return t
    return t * TensorInterface.onehot(scalartype(t), only(aux) => 1)
end
_value_sum(t) = sum(_value_slice(t))

# BP normalises each message by the sum of ALL its entries; with the non-Hermitian `a > 0`
# half-insertion slices that sum is complex and the whole message picks up a phase, which makes
# the norm sector non-Hermitian and H_eff/N_eff below non-Hermitian maps. Renormalise by the
# norm-sector sum instead (a gauge choice: Z_B is invariant under rescaling any message).
function _hermitian_gauge!(bpc::BeliefPropagationCache)
    for e in collect(keys(messages(bpc)))
        m = message(bpc, e)
        setmessage!(bpc, e, m / _value_sum(m))
    end
    return bpc
end

_ket_tensor(bpc::BeliefPropagationCache, v) = ket(network(bpc))[v]
_bra_tensor(bpc::BeliefPropagationCache, v) = bra_tensor(network(bpc), v)
_incoming_edges(bpc::BeliefPropagationCache, v) = NamedGraphs.GraphsExtensions.boundary_edges(bpc, [v]; dir = :in)

"""
    bethe_energy(bpc::BeliefPropagationCache, gen::GeneratingOperator)

The Bethe energy `∂λ Z_B(T(λ), X₀)|₀ / Z_B` of the generating cache `bpc` (envelope theorem: the
messages are held fixed). The edge scalars carry no explicit λ, so it is `Σᵥ ∂λZᵥ / Zᵥ` with the
vertex operator replaced by its λ-derivative in the numerator. Equals the BP expectation value of
`H` at χ = 1 and the exact energy on a tree.
"""
function bethe_energy(bpc::BeliefPropagationCache, gen::GeneratingOperator)
    E = zero(scalartype(network(bpc)))
    for v in vertices(bpc)
        ms = incoming_messages(bpc, v)
        num = _region_scalar(vcat([_ket_tensor(bpc, v), gen.derivative[v], _bra_tensor(bpc, v)], ms))
        den = _region_scalar(vcat([_ket_tensor(bpc, v), gen.value[v], _bra_tensor(bpc, v)], ms))
        E += num / den
    end
    return E
end

"""
    bethe_energy(ψ::TensorNetworkState, H::Vector; kwargs...)

Convenience: build the generating operator of the Hamiltonian terms `H`, converge BP on the
generating norm network (`kwargs` to `update`) and return the Bethe energy.
"""
function bethe_energy(ψ::TensorNetworkState, H::Vector; kwargs...)
    gen = generating_operator(H, ψ)
    return bethe_energy(generating_cache(ψ, gen; kwargs...), gen)
end

# ── Linear response of the messages ──────────────────────────────────────────────────────
"""
    message_response(bpc::BeliefPropagationCache, gen::GeneratingOperator; maxiter = 100, tol = 1e-10, verbose = false)

`X′ = dX/dλ|₀`: the first-order response of the BP fixed point to the Hamiltonian source, from the
linearised normalised message update `m = F̃(X)/ΣF̃(X)`:

    dm = (dF̃ − m ΣdF̃) / ΣF̃,   dF̃ = F̃ with the vertex operator → ∂λG  +  Σ_w F̃ with m_{w→u} → dm_{w→u}

solved by Gauss–Seidel sweeps in the cache's edge sequence (exact after one sweep on a tree; a
contraction whenever BP itself is stable). Returns a `Dictionary` over directed edges.
"""
function message_response(bpc::BeliefPropagationCache, gen::GeneratingOperator; maxiter::Int = 100, tol::Real = 1.0e-10, verbose::Bool = false, history = nothing)
    es = edge_sequence(bpc)
    dX = Dictionary{NamedEdge, Any}()
    for e in es
        set!(dX, e, TensorInterface.scale!(copy(message(bpc, e)), 0))
    end
    # per directed edge: incoming edges, fixed factors and a contraction sequence shared by all
    # single-message replacements (identical index structure)
    plan = Dictionary{NamedEdge, Any}()
    for e in es
        u = src(e)
        inc = filter(!=(reverse(e)), _incoming_edges(bpc, u))
        ms = [message(bpc, ie) for ie in inc]
        F̃ = _region_scalar_tensor(vcat([_ket_tensor(bpc, u), gen.value[u], _bra_tensor(bpc, u)], ms), nothing)
        seq = contraction_sequence(vcat([_ket_tensor(bpc, u), gen.value[u], _bra_tensor(bpc, u)], ms); alg = "optimal")
        set!(plan, e, (inc, ms, _value_sum(F̃), seq))
    end
    diff = Inf
    for it in 1:maxiter
        diff = 0.0
        for e in es
            u = src(e)
            inc, ms, s, seq = plan[e]
            m = message(bpc, e)
            dF = contract(vcat([_ket_tensor(bpc, u), gen.derivative[u], _bra_tensor(bpc, u)], ms); sequence = seq)
            for (k, ie) in enumerate(inc)
                ms2 = copy(ms); ms2[k] = dX[ie]
                dF = dF + contract(vcat([_ket_tensor(bpc, u), gen.value[u], _bra_tensor(bpc, u)], ms2); sequence = seq)
            end
            dm = (dF - m * _value_sum(dF)) / s     # norm-sector normalisation gauge (real): keeps dm Hermitian
            diff = max(diff, norm(dm - dX[e]) / max(norm(dm), eps()))
            set!(dX, e, dm)
        end
        verbose && println("response sweep $it: max relative change $diff")
        history === nothing || push!(history, diff)
        diff < tol && break
    end
    diff < tol || @warn "message_response: not converged to $tol (last change $diff)"
    return dX
end
_region_scalar_tensor(ts::Vector, ::Nothing) = contract(ts; sequence = contraction_sequence(ts; alg = "optimal"))

# ── Local effective operators ────────────────────────────────────────────────────────────
"""
    effective_operators(bpc, gen, dX, v)

Matrix-free `N_eff` and `H_eff` at vertex `v` as functions of the ket tensor (output on the ket's
own indices): `N_eff ψ = ψ · G(0) · X₀`, `H_eff ψ = ψ · ∂λG · X₀ + Σᵤ ψ · G(0) · X₀[m_{u→v} → X′_{u→v}]`,
the bra tensor removed. Up to a multiple of `N_eff` (a λ-derivative of the rest of the network,
which shifts the generalised eigenvalue but not the eigenvector) this is the gradient pair of the
Bethe energy with respect to `ψᵥ*`.
"""
function effective_operators(bpc::BeliefPropagationCache, gen::GeneratingOperator, dX, v)
    inc = _incoming_edges(bpc, v)
    ms = [message(bpc, e) for e in inc]
    G0, dG = gen.value[v], gen.derivative[v]
    ψv = _ket_tensor(bpc, v)
    seq = contraction_sequence(vcat([ψv, G0], ms); alg = "optimal")
    N(x) = TensorInterface.noprime(contract(vcat([x, G0], ms); sequence = seq))
    function H(x)
        y = contract(vcat([x, dG], ms); sequence = seq)
        for (k, e) in enumerate(inc)
            ms2 = copy(ms); ms2[k] = dX[e]
            y = y + contract(vcat([x, G0], ms2); sequence = seq)
        end
        return TensorInterface.noprime(y)
    end
    return N, H
end

# The norm environment factorises over the incoming bonds, N_eff = ⊗ᵤ Mᵤ ⊗ 1_site with Mᵤ the
# a = 0 (norm) component of the message; its square roots are the simple-update gauge tensors.
function _norm_roots(bpc::BeliefPropagationCache, v; cutoff = nothing)
    roots = Any[]; iroots = Any[]
    for e in _incoming_edges(bpc, v)
        m = message(bpc, e)
        aux = filter(i -> occursin("aux", string(TensorInterface.tags(i))), collect(TensorInterface.inds(m)))
        M = isempty(aux) ? m : m * TensorInterface.onehot(scalartype(m), only(aux) => 1)
        r, ir = pseudo_sqrt_inv_sqrt(M; cutoff = isnothing(cutoff) ? defaulttol(M) : cutoff)
        push!(roots, r); push!(iroots, ir)
    end
    return roots, iroots
end
_apply_bond_maps(x, maps) = TensorInterface.noprime(foldl((t, M) -> t * M, maps; init = x))

"""
    optimize_vertex!(bpc, gen, dX, v; krylov_kwargs...)

Solve the local generalised eigenproblem `H_eff ψ = E N_eff ψ` at `v` for the lowest eigenvalue
by whitening with the message square roots (`H̃ = N⁻¹ᐟ² H_eff N⁻¹ᐟ²`, Hermitian) and a Lanczos
solve, and write the new tensor into the cache's ket (normalised to `⟨ψᵥ|N_eff|ψᵥ⟩ = 1`). The
messages are NOT updated here. Returns the local eigenvalue.
"""
function optimize_vertex!(bpc::BeliefPropagationCache, gen::GeneratingOperator, dX, v;
                          tol::Real = 1.0e-10, krylovdim::Int = 12, maxiter::Int = 20, sqrt_cutoff = nothing)
    N, H = effective_operators(bpc, gen, dX, v)
    roots, iroots = _norm_roots(bpc, v; cutoff = sqrt_cutoff)
    ψv = _ket_tensor(bpc, v)
    is = collect(TensorInterface.inds(ψv))
    flat(t) = vec(TensorInterface.array(t, is...))
    unflat(x) = TensorInterface.from_array(reshape(x, TensorInterface.dim.(is)...), is...)
    H̃(x) = flat(_apply_bond_maps(H(_apply_bond_maps(unflat(x), iroots)), iroots))
    x0 = flat(_apply_bond_maps(ψv, roots))
    vals, vecs, info = eigsolve(H̃, x0, 1, :SR; ishermitian = true, tol, krylovdim, maxiter)
    info.converged ≥ 1 || @warn "optimize_vertex!: Lanczos not converged at $v (residual $(info.normres))"
    ψnew = _apply_bond_maps(unflat(vecs[1]), iroots)
    setindex_preserve!(ket(network(bpc)), ψnew, v)
    return real(vals[1])
end

"""
    dmrg(ψ::TensorNetworkState, H::Vector; nsweeps = 5, bp_kwargs = (;), response_kwargs = (;), krylov_kwargs = (;), verbose = true)

One-site variational ground-state search on the bond dimension of `ψ`: for every vertex in turn
solve the local generalised eigenproblem of the Bethe energy (norm environment `N_eff`, effective
Hamiltonian `H_eff` including the environment response), re-converge the generating norm network
and its response, and record the Bethe energy. Exact DMRG on a tree; the BP (χ = 1) approximation
on a loopy graph. Returns `(ψ, energies)` with one energy per vertex update.
"""
function dmrg(ψ::TensorNetworkState, H::Vector; nsweeps::Int = 5, bp_kwargs = (;), response_kwargs = (;),
              krylov_kwargs = (;), verbose::Bool = true, vertex_order = collect(vertices(ψ)))
    gen = generating_operator(H, ψ)
    bpc = generating_cache(copy(ψ), gen; bp_kwargs...)
    energies = Float64[]
    for sweep in 1:nsweeps
        for v in vertex_order
            dX = message_response(bpc, gen; response_kwargs...)
            optimize_vertex!(bpc, gen, dX, v; krylov_kwargs...)
            bpc = _hermitian_gauge!(update(bpc; bp_kwargs...))
            push!(energies, real(bethe_energy(bpc, gen)))
            verbose && println("sweep $sweep, vertex $v: E_B = $(last(energies))")
        end
    end
    return ket(network(bpc)), energies
end
