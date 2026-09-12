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
    dmrg(ψ::TensorNetworkState, H::Vector; alg = "bp", kwargs...)

One-site variational ground-state search on the bond dimension of `ψ` for the Hamiltonian term
list `H`, with the environments of

- `alg = "bp"` (default): belief propagation (χ = 1). `kwargs`: `nsweeps = 5`, `bp_kwargs`,
  `response_kwargs`, `krylov_kwargs`, `verbose`, `vertex_order`. Exact DMRG on a tree, the BP
  approximation on a loopy graph.
- `alg = "ctmrg"`: finite-CTMRG (matrix-product BP) rings of interface dimension `maxdim`
  (required) on a 2D grid; see [`dmrg(::Algorithm"ctmrg", ...)`](@ref) for the keywords.

Returns `(ψ, energies)` with one energy per vertex update.
"""
dmrg(ψ::TensorNetworkState, H::Vector; alg = "bp", kwargs...) = dmrg(Algorithm(alg), ψ, H; kwargs...)

# χ = 1: for every vertex in turn solve the local generalised eigenproblem of the Bethe energy
# (norm environment `N_eff`, effective Hamiltonian `H_eff` including the environment response),
# re-converge the generating norm network and its response, and record the Bethe energy.
function _dmrg_bp(ψ::TensorNetworkState, H::Vector; nsweeps::Int = 5, bp_kwargs = (;), response_kwargs = (;),
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

# ══════════════════════════════════════════════════════════════════════════════════════════
# CTM (matrix-product BP) environments — phase 2
#
# The same construction with the χ = 1 messages replaced by the finite-CTMRG rings of the
# generating norm network ⟨ψ|G(λ)|ψ⟩. The rings are `CTMEnvironmentCache` environments of a
# `QuadraticForm` whose operator layer is the generating operator, so the three-layer factor
# list [ket, G, bra] runs through the corner moves unchanged (`bp_factors` of an `AbstractForm`).
#
# What differs from χ = 1 is how the environment RESPONSE dX/dλ enters. At χ = 1 it is a
# linear solve on the message tangent space. Here it is never formed: the effective ring
#
#     E_λ ψᵥ = ring_λ · G(λ)ᵥ · ψᵥ
#
# is a CLOSED contraction over every truncated interface, hence invariant under the interface
# gauge (the biorthogonal projector pairs, their sweep-to-sweep rotations), and its λ-derivative
# is taken by a central finite difference of the rings re-converged at ±λ, warm-started from
# λ = 0. No Hessian, no Jacobian-vector product of the Schur solve, no tangent gauge fixing.
# The energy itself never sees the finite difference: it is the envelope-theorem derivative at
# fixed rings (`bethe_energy`), and a finite-difference error in H_eff perturbs the variational
# energy only at second order. Measured (docs/dmrg.md, 2026-09-11): with `:cycle` and λ = 1e-6
# the gradient of the CTM energy agrees with the exact-energy derivative to 5e-6 at χ = 32 on
# a 4×4 D = 3 TFIM state, where `:cut` at the same χ is off by 2e-2.
#
# `:cycle` needs a SMALL λ (1e-6): at λ ≥ 1e-5 the warm-started ±λ solves leave the basin of
# the λ = 0 fixed point and land on a different invariant subspace (measured: gradient 10–50%
# off, full cold-start cost). The finite-difference cancellation at 1e-6 is ~1e-10 relative
# because the ring blocks are norm-rescaled, so there is a comfortable window.
# ══════════════════════════════════════════════════════════════════════════════════════════

# G(0) + λ ∂λG in the same representation — the operator layer of T(λ) to first order, which is
# all a central difference at λ = 0 sees.
function _shifted_operator(gen::GeneratingOperator, λ::Real)
    vs = collect(vertices(gen.value))
    ts = Dictionary(vs, [gen.value[v] + λ * gen.derivative[v] for v in vs])
    return TensorNetworkOperator(ts, copy(graph(gen.value)), copy(siteinds(gen.value)))
end

"""
    generating_cache(ψ, gen, maxdim; λ = 0, seed = nothing, projector = :cycle, kwargs...) -> CTMEnvironmentCache

Converged finite-CTMRG environments (`4C + 4T` ring per vertex, interfaces truncated to
`maxdim`) of the generating norm network `⟨ψ|G(λ)|ψ⟩`. `seed` warm-starts the sweep from
another cache's environments (same state indices); `projector` and the remaining `kwargs`
(`maxiter`, `tolerance`, `convergence`) go to [`CTMEnvironmentCache`](@ref) and [`update`](@ref).
"""
function generating_cache(ψ::TensorNetworkState, gen::GeneratingOperator, maxdim::Integer; λ::Real = 0,
                          seed = nothing, projector::Symbol = :cycle, convergence::Symbol = :marginal,
                          maxiter::Integer = 100, tolerance::Real = 1.0e-12, kwargs...)
    operator = iszero(λ) ? gen.value : _shifted_operator(gen, λ)
    cache = CTMEnvironmentCache(QuadraticForm(ψ, operator), maxdim; projector, kwargs...)
    seed === nothing || (cache = _ctm_setenv(cache, environments(seed)))
    return update(cache; maxiter, tolerance, convergence)
end

"""
    bethe_energy(cache::CTMEnvironmentCache, gen::GeneratingOperator)

The energy `∂λ Z_B(T(λ), X₀)|₀ / Z_B` of the CTM (CVM) functional at fixed rings — the envelope
theorem — as `Σᵥ ⟨ringᵥ · ∂λGᵥ⟩ / ⟨ringᵥ · G(0)ᵥ⟩`. Only the vertex regions carry λ explicitly
(edge and plaquette regions are environment blocks alone), and the per-block rescaling cancels in
each ratio. O(ε²) in the truncation error under the stationary `:cycle` projector, O(ε) under `:cut`.
"""
function bethe_energy(cache::CTMEnvironmentCache, gen::GeneratingOperator; window::Integer = 0)
    qf = network(cache)
    qf isa QuadraticForm || error("bethe_energy: the CTM cache must wrap a QuadraticForm over the generating operator")
    ψ = ket(qf); opts = options(cache)
    E = zero(real(scalartype(ψ)))
    for v in vertices(ψ)
        # `window = 1` keeps the 3×3 patch exact and removes the `:cycle` edge plateau: a bond
        # term whose partner sits in a rank-capped two-site boundary block is otherwise read
        # through that block's truncated interface, at a χ-independent 4.5e-9 per such vertex on
        # the 4×4 D = 3 TFIM (docs/dmrg.md, overnight 2026-09-11/12).
        ring = vertex_window(cache, v, window)
        num = scalar(_ctm_contract(vcat(ring, [ψ[v], gen.derivative[v], bra_tensor(qf, v)]), opts))
        den = scalar(_ctm_contract(vcat(ring, [ψ[v], gen.value[v], bra_tensor(qf, v)]), opts))
        E += real(num / den)
    end
    return E
end

# Dense effective ring at `v`: `ring · opv` as a matrix from `ψᵥ`'s index set to its primed copy.
function _dense_ring_operator(ring::Vector, opv, ψv, opts::CTMOptions)
    E = _ctm_contract(vcat(ring, [opv]), opts)
    is = collect(inds(ψv))
    A = TensorInterface.array(E, prime.(is)..., is...)
    n = prod(TensorInterface.dim.(is))
    return reshape(A, n, n)
end

"""
    effective_operators(cache, cache_plus, cache_minus, gen, v, λ) -> (N_eff, H_eff)

Dense `N_eff` and `H_eff` at `v` from the λ = 0 ring and the central finite difference of the
effective rings at ±λ (`generating_cache(ψ, gen, χ; λ = ±λ, seed = cache)`), in the basis of
`ψ[v]`'s indices. Both are symmetrised. `H_eff` carries an arbitrary multiple of `N_eff` (the
λ-derivative of the block rescaling), which shifts the generalised eigenvalue but not the
eigenvector: the local Rayleigh quotient is NOT the energy, [`bethe_energy`](@ref) is.
"""
function effective_operators(cache::CTMEnvironmentCache, cache_plus::CTMEnvironmentCache,
                             cache_minus::CTMEnvironmentCache, gen::GeneratingOperator, v, λ::Real)
    opts = options(cache); ψv = ket(network(cache))[v]
    G0, dG = gen.value[v], gen.derivative[v]
    N = _dense_ring_operator(vertex_ring(cache, v), G0, ψv, opts)
    Hp = _dense_ring_operator(vertex_ring(cache_plus, v), G0 + λ * dG, ψv, opts)
    Hm = _dense_ring_operator(vertex_ring(cache_minus, v), G0 - λ * dG, ψv, opts)
    H = (Hp - Hm) / (2λ)
    return (N + N') / 2, (H + H') / 2
end

"""
    optimize_vertex!(ψ, v, N, H; whiten_cutoff = 1e-6) -> eigenvalue

Lowest generalised eigenvector of `H ψᵥ = E N ψᵥ` by whitening (`N = U S U†`, directions with
`S < whiten_cutoff · S_max` dropped) and a dense Hermitian eigensolve, written into `ψ[v]`
normalised to unit Frobenius norm. The cutoff is NOT a tolerance: a bond direction the state
barely uses has an `N_eff` weight down at 1e-11 and an `H_eff` there that is pure truncation
noise, and keeping it (cutoff 1e-12) measured as a monotone energy RISE of 3e-3 over a sweep on
a 4×4 D = 3 state, where 1e-6 descends cleanly. Returns the eigenvalue (see
[`effective_operators`](@ref) for why it is not the energy).
"""
function optimize_vertex!(ψ::TensorNetworkState, v, N::AbstractMatrix, H::AbstractMatrix;
                          whiten_cutoff::Real = 1.0e-6, damping::Real = 0)
    S, U = eigen(Hermitian(N))
    keep = S .> whiten_cutoff * maximum(S)
    W = U[:, keep] * Diagonal(S[keep] .^ -0.5)
    vals, vecs = eigen(Hermitian(W' * H * W))
    T = scalartype(ψ[v])
    is = collect(inds(ψ[v]))
    x0 = vec(TensorInterface.array(ψ[v], is...))
    # On a graded site the dense problem is block-diagonal over the tensor's total charge, and the
    # lowest eigenvector may sit in a block the state does not occupy; `from_array` projects onto
    # the state's block, so take the lowest eigenvector that survives the projection.
    local ψnew, val
    for j in 1:length(vals)
        x = W * vecs[:, j]
        T <: Real && (x = real(x))        # the operator tensors are complex; a real state stays real
        x = x / norm(x)
        if !iszero(damping)               # mix with the old tensor, phase-aligned, then renormalise
            s = dot(x0, x); x = (1 - damping) * (s == 0 ? x : x * sign(s)) + damping * (x0 / norm(x0))
        end
        ψnew = TensorInterface.from_array(reshape(Vector{T}(x), TensorInterface.dim.(is)...), is...)
        val = vals[j]
        norm(ψnew) > 0.5 && break
    end
    ψ[v] = ψnew / norm(ψnew)
    return real(val)
end

# The Vidal/BP gauge with the ORIGINAL bond indices restored, so environments keyed on them can
# still seed the next sweep. One-site updates put arbitrary weight on the bonds, and neither
# projector's truncation is uniform in the gauge (`:cut` is not gauge equivariant at all).
function _regauge_keep_inds(ψ::TensorNetworkState)
    ψg = gauge_and_scale(ψ)
    for e in edges(ψ)
        old, new = virtualinds(ψ, e), virtualinds(ψg, e)
        for v in (src(e), dst(e))
            ψg[v] = replaceinds(ψg[v], new, old)
        end
    end
    return ψg
end

"""
    dmrg(ψ, H; alg = "ctmrg", maxdim, nsweeps = 2, projector = :cut, λ = nothing,
         whiten_cutoff = 1e-6, regauge = false, ctm_kwargs = (;), verbose = true)

One-site ground-state sweeps with finite-CTMRG (matrix-product BP) environments of interface
dimension `maxdim`: at every vertex the λ = 0 rings are re-converged (warm-started), the ±λ
rings give `H_eff` by finite difference, the local generalised eigenproblem is solved densely,
and (with `regauge = true`) the state is put back in the Vidal gauge. Returns `(ψ, energies)`
with one [`bethe_energy`](@ref) per vertex update. Dense (non-graded) states only for now.

`projector = :cut` is the default although `:cycle` gives the ε² energy and a far more accurate
gradient on a fixed state (docs/dmrg.md): once the state moves, the `:cycle` solves fall out of
their warm-start basin and a sweep costs minutes per vertex against seconds for `:cut`. `λ`
defaults to the measured window per projector (1e-6 for `:cycle`, 1e-7 for `:cut`).

`refresh = :vertex` re-converges the three environments after every vertex (one energy per vertex
update). `:checkerboard` / `:fourcolour` update a parity / (x mod 2, y mod 2) class against the
same environments and refresh per class; `:sweep` refreshes once per sweep. Grouped refreshes are
`Lx·Ly` times cheaper but overshoot — measured divergent undamped on the 4×4 D = 3 (`:sweep` at
once, `:checkerboard` by the third refresh, `:fourcolour` by the ninth) and stable with
`damping = 0.5`. Every refresh is an ACCEPTANCE step: an update that raises the energy by more than
`accept_tol` (relative) is reverted and counted, with a warning at the end.

`energy = :fd` (default) records `(F(+λ) − F(−λ)) / 2λ`, the implicit derivative with the
environments re-converged, which under `:cut` at truncated χ is 100–500× more accurate than the
fixed-ring `energy = :ring` ([`bethe_energy`](@ref)); `:window` is the fixed-ring energy over the
exact 3×3 window, which removes the `:cycle` edge plateau. Under `:cycle`, `cycle_gapcut = 1e-4` is
put into `ctm_kwargs` unless given (it keeps the ±λ solves in the basin of the λ = 0 fixed point).
"""
function dmrg(::Algorithm"ctmrg", ψ::TensorNetworkState, H::Vector; maxdim::Integer, nsweeps::Int = 2,
              projector::Symbol = :cut, λ::Union{Real, Nothing} = nothing, whiten_cutoff::Real = 1.0e-6,
              regauge::Bool = false, ctm_kwargs = (;), verbose::Bool = true,
              vertex_order = collect(vertices(ψ)), refresh::Symbol = :vertex, energy::Symbol = :fd,
              damping::Real = 0, accept_tol::Real = 1.0e-9)
    refresh in (:vertex, :sweep, :checkerboard, :fourcolour) || throw(ArgumentError(
        "refresh must be :vertex, :sweep, :checkerboard or :fourcolour, got $(repr(refresh))"))
    energy in (:ring, :window, :fd) || throw(ArgumentError("energy must be :ring, :window or :fd, got $(repr(energy))"))
    # Finite-difference step. Measured window (docs/dmrg.md): `:cycle` needs λ ≤ 1e-6 to keep the
    # ±λ warm starts in the basin of the λ = 0 fixed point, `:cut` needs λ ≤ 1e-7 on random states
    # (1e-5 is 10% off there, 4e-5 on a physical state); below 1e-8 the cancellation error shows.
    λ = something(λ, projector === :cycle ? 1.0e-6 : 1.0e-7)
    # `:cycle` on the generating network has rank-capped boundary interfaces whose surplus null
    # modes wander between the λ = 0 and ±λ solves; the noise-cliff cut removes them. Measured
    # (docs/dmrg.md, overnight): gradient error at λ = 1e-5 0.107 → 2.2e-6 with `cycle_gapcut = 1e-4`.
    if projector === :cycle && !haskey(ctm_kwargs, :cycle_gapcut)
        ctm_kwargs = (; cycle_gapcut = 1.0e-4, ctm_kwargs...)
    end
    ψ = copy(ψ)
    gen = generating_operator(H, ψ)
    cache = generating_cache(ψ, gen, maxdim; projector, ctm_kwargs...)
    energies = Float64[]
    nreject = 0
    # The three environments of the current state. The ±λ caches also give the FD-of-F energy,
    # `(F(+λ) − F(−λ)) / 2λ` — d/dλ ln Ẑ with the environments re-converged, which carries the
    # projector response and is measured far more accurate than the fixed-ring `bethe_energy` under
    # `:cut` at truncated χ (docs/dmrg.md, H1). `:window` is the fixed-ring energy over the exact
    # 3×3 window, which removes the `:cycle` edge plateau (same doc).
    function environments!(cache)
        cp = generating_cache(ψ, gen, maxdim; λ = λ, seed = cache, projector, ctm_kwargs...)
        cm = generating_cache(ψ, gen, maxdim; λ = -λ, seed = cache, projector, ctm_kwargs...)
        E = energy === :fd ? (cvm_freenergy(cp) - cvm_freenergy(cm)) / (2λ) :
            bethe_energy(cache, gen; window = energy === :window ? 1 : 0)
        return cp, cm, E
    end
    cp, cm, E = environments!(cache)
    # Vertices updated against the SAME environments before a refresh. `:vertex` is Gauss–Seidel;
    # `:sweep` (Jacobi) measured divergent on the 4×4 D = 3 (E rose from the first sweep and blew
    # up by the third); `:checkerboard` refreshes twice per sweep, after each parity class, so no
    # two vertices updated together are neighbours.
    groups = if refresh === :vertex
        [[v] for v in vertex_order]
    elseif refresh === :sweep
        [vertex_order]
    elseif refresh === :checkerboard
        par(v) = isodd(sum(Int, v))
        [filter(!par, vertex_order), filter(par, vertex_order)]
    else                                   # :fourcolour — (x mod 2, y mod 2) classes
        cls(v) = (mod(Int(v[1]), 2), mod(Int(v[2]), 2))
        [filter(v -> cls(v) == c, vertex_order) for c in ((0, 0), (1, 1), (0, 1), (1, 0))]
    end
    for sweep in 1:nsweeps
        for group in groups
            old = Dict(v => ψ[v] for v in group)
            for v in group
                N, Heff = effective_operators(cache, cp, cm, gen, v, λ)
                optimize_vertex!(ψ, v, N, Heff; whiten_cutoff, damping)
            end
            regauge && (ψ = _regauge_keep_inds(ψ))
            cache_new = generating_cache(ψ, gen, maxdim; seed = cache, projector, ctm_kwargs...)
            cp_new, cm_new, E_new = environments!(cache_new)
            # ACCEPTANCE: a variational step must not raise the energy. One that does means H_eff
            # was wrong in the direction taken (a near-null N_eff direction, an overshoot of a
            # grouped update), and letting it through is how the 4×4 D = 3 runs blew up from a
            # 6e-5 gap to E = −38 in one step. Revert the group and keep the old environments.
            if E_new > E + accept_tol * max(one(E), abs(E))
                nreject += 1
                for (v, t) in old
                    ψ[v] = t
                end
                verbose && println("sweep $sweep: rejected update of $(length(group) == 1 ? "vertex $(only(group))" : "$(length(group)) vertices") (E would rise to $E_new from $E)")
            else
                cache, cp, cm, E = cache_new, cp_new, cm_new, E_new
                verbose && println("sweep $sweep, after $(length(group) == 1 ? "vertex $(only(group))" : "$(length(group)) vertices"): E = $E")
            end
            push!(energies, E)
        end
    end
    nreject > 0 && @warn "dmrg (ctmrg): $nreject of $(nsweeps * length(groups)) updates were rejected because they raised the energy; " *
                         "consider a larger `maxdim`, a stricter `whiten_cutoff`, or `damping` for grouped refreshes."
    return ψ, energies
end

dmrg(::Algorithm"bp", ψ::TensorNetworkState, H::Vector; kwargs...) = _dmrg_bp(ψ, H; kwargs...)
