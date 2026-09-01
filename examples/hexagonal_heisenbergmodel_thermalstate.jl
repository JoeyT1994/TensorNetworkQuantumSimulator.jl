"""
Finite-temperature antiferromagnetic Heisenberg purification on a finite open hexagonal lattice,
followed by a matched-boundary-rank comparison of boundary MPS, SVD-CTMRG (`:cut`), and
eig-CTMRG (`:cycle`) for both `ln Z` and the site-averaged staggered magnetisation.

The purification applies `exp(-dβ H)` to one leg of the infinite-temperature identity, so the
norm represents `Z(β)` at `β = 2 nsteps dβ`, matching the convention of the original example.
On a finite spin-symmetric system `⟨Mₛ⟩` is identically zero. The benchmark therefore applies a
weak staggered pinning field `hₛ` during imaginary-time evolution and measures
`mₛ = N⁻¹ Σᵢ ηᵢ ⟨Sᵢᶻ⟩`, where `ηᵢ = ±1` is the honeycomb bipartition. Set
`HEX_THERMAL_STAGGERED_FIELD=0` to recover the unpinned model.

The default run is intentionally small. A useful larger run is, for example:

    HEX_THERMAL_NX=3 HEX_THERMAL_NY=3 HEX_THERMAL_D=4 \
    HEX_THERMAL_DBETA=0.02 HEX_THERMAL_STEPS=25 \
    HEX_THERMAL_STAGGERED_FIELD=0.05 \
    HEX_THERMAL_CHIS=4,8,12,16,24 HEX_THERMAL_REF_CHI=48 \
    julia --project=. examples/hexagonal_heisenbergmodel_thermalstate.jl

The CSV contains each returned value, its error against the high-χ bMPS reference, the change
from the preceding χ for the same method, runtime, and CTMRG marginal inconsistency.
"""

using TensorNetworkQuantumSimulator
using ITensors: ITensors, ITensor, contract, scalar
using Logging: SimpleLogger, Warn, with_logger
using Printf
using Statistics: mean
using Dictionaries: Dictionary

const TNQS = TensorNetworkQuantumSimulator

envint(name, default) = parse(Int, get(ENV, name, string(default)))
envfloat(name, default) = parse(Float64, get(ENV, name, string(default)))
envbool(name, default) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes")
envints(name, default) = parse.(Int, split(get(ENV, name, join(default, ',')), ','))

function staggered_signs(g)
    vs = collect(vertices(g))
    isempty(vs) && return Dict{Any, Int}()
    signs = Dict{Any, Int}(first(vs) => 1)
    queue = Any[first(vs)]
    while !isempty(queue)
        v = popfirst!(queue)
        for e in edges(g)
            u = src(e) == v ? dst(e) : dst(e) == v ? src(e) : nothing
            isnothing(u) && continue
            if haskey(signs, u)
                signs[u] == -signs[v] || error("Honeycomb graph is not bipartite at edge $e")
            else
                signs[u] = -signs[v]
                push!(queue, u)
            end
        end
    end
    length(signs) == length(vs) || error("Honeycomb graph is disconnected")
    return signs
end

function edge_coupling(couplings, e)
    couplings isa Number && return couplings
    haskey(couplings, e) && return couplings[e]
    haskey(couplings, reverse(e)) && return couplings[reverse(e)]
    error("Missing Heisenberg coupling for edge $e")
end

function heisenberg_layer(s, g, couplings, dβ, staggered_field, signs)
    gates = ITensor[]
    # Rxxyyzz(θ) = exp[-i θ (XX+YY+ZZ)/2]. With θ=-i J dβ/2 this is
    # exp[-dβ J (XX+YY+ZZ)/4] = exp[-dβ J S⋅S]. The three edge colors are
    # non-overlapping matchings and form one first-order Trotter step.
    for matching in edge_color(g, 3)
        append!(gates, [ITensors.op(
            "Rxxyyzz", s[src(e)][1], s[dst(e)][1];
            θ = -0.5 * edge_coupling(couplings, e) * dβ * im
        ) for e in matching])
    end
    # H_field = -hₛ Σᵢ ηᵢ Sᵢᶻ. Since ITensors' `Z` is 2Sᶻ and the purification
    # evolves only its ket leg, exp(-dβ H_field) = exp(+dβ hₛ ηᵢ Zᵢ/2).
    if !iszero(staggered_field)
        append!(gates, [exp(0.5dβ * staggered_field * signs[v] *
                            ITensors.op("Z", s[v][1])) for v in vertices(g)])
    end
    return gates
end

function prepare_thermal_state(; nx, ny, D, J, dβ, nsteps, cutoff, staggered_field)
    # Open boundaries are essential here: both finite CTMRG and boundary MPS contract this one
    # finite graph. The old periodic 2×2 example was a BP unit-cell calculation and is not a
    # meaningful head-to-head finite-contraction benchmark.
    g = named_hexagonal_lattice_graph(nx, ny; periodic = false)
    s = siteinds("S=1/2", g; inds_per_site = 2)
    ψ = identity_tensornetworkstate(Float64, g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    signs = staggered_signs(g)
    layer = heisenberg_layer(s, g, J, dβ, staggered_field, signs)
    apply_kwargs = (; maxdim = D, cutoff, normalize_tensors = false)

    # Keep an explicit ledger of every extensive scalar removed by `rescale!`. This is the
    # normalization convention of the original example. All three contractors below see the same
    # well-scaled residual network, and the common ledger is added back to their residual lnZ.
    accumulated_log_scale = real(TNQS.freenergy(ψ_bpc))
    rescale!(ψ_bpc)
    max_gate_error = 0.0
    for step in 1:nsteps
        elapsed = @elapsed ψ_bpc, errors = apply_gates(
            layer, ψ_bpc; apply_kwargs, verbose = false)
        layer_error = isempty(errors) ? 0.0 : maximum(real.(errors))
        max_gate_error = max(max_gate_error, layer_error)
        removed = real(TNQS.freenergy(ψ_bpc))
        accumulated_log_scale += removed
        rescale!(ψ_bpc)
        @printf("imaginary-time step %d/%d: β=%.4f  D=%d  max gate error=%.3e  ΔlnZscale=%.6e  %.2f s\n",
                step, nsteps, 2step * dβ, maxvirtualdim(ψ_bpc), layer_error, removed,
                elapsed)
    end
    return network(ψ_bpc), g, signs, max_gate_error, accumulated_log_scale
end

function double_layer_tensor(state, v; op = "I")
    t = state[v]
    sinds = siteinds(state, v)
    length(sinds) == 2 || error(
        "Expected physical and ancilla indices at $v; got $(length(sinds)) site indices")
    physical, ancilla = sinds
    tdag = ITensors.dag(ITensors.prime(t))
    if op == "I"
        tdag = ITensors.replaceinds(tdag, ITensors.prime.(sinds), sinds)
        return t * tdag
    end
    tdag = ITensors.replaceinds(tdag, [ITensors.prime(ancilla)], [ancilla])
    return t * tdag * ITensors.op(op, physical)
end

function double_layer_network(state, g)
    vs = collect(vertices(g))
    plain = Dictionary(vs, [double_layer_tensor(state, v) for v in vs])
    zinsert = Dictionary(vs, [double_layer_tensor(state, v; op = "Z") for v in vs])
    return TNQS.TensorNetwork(plain, g), plain, zinsert
end

function staggered_magnetisation_bmps(cache, plain, zinsert, signs)
    vs = collect(vertices(graph(cache)))
    local_cache = TNQS.update_partitions(cache, vs)
    values = map(vs) do v
        env = TNQS.incoming_messages(local_cache, v)
        numerator = scalar(contract(ITensor[zinsert[v]; env]))
        denominator = scalar(contract(ITensor[plain[v]; env]))
        0.5 * signs[v] * real(numerator / denominator)
    end
    return mean(values)
end

function staggered_magnetisation_ctm(cache, plain, zinsert, signs)
    values = map(collect(vertices(graph(cache)))) do v
        env = vertex_ring(cache, v)
        numerator = scalar(TNQS._ctm_contract(ITensor[zinsert[v]; env], TNQS.options(cache)))
        denominator = scalar(TNQS._ctm_contract(ITensor[plain[v]; env], TNQS.options(cache)))
        0.5 * signs[v] * real(numerator / denominator)
    end
    return mean(values)
end

function contract_bmps(tn, plain, zinsert, signs, χ)
    local cache
    elapsed = @elapsed cache = update(BoundaryMPSCache(
        tn, χ; partition_by = "row", gauge_state = false))
    staggered_magnetisation = staggered_magnetisation_bmps(cache, plain, zinsert, signs)
    return log(abs(partitionfunction(cache))), staggered_magnetisation, NaN, elapsed, true
end

function contract_ctm(tn, plain, zinsert, signs, χ, projector;
                      maxiter, tolerance, verbose)
    initial = CTMEnvironmentCache(tn, χ; projector, gauge_state = false)
    local cache
    # Capture (and then replay) warnings so the CSV records whether update reached its requested
    # stopping criterion. The returned scalar can already be converged when over-parametrized null
    # directions keep the state-distance guard moving, so value accuracy and solver certification
    # are deliberately separate columns.
    warning_buffer = IOBuffer()
    elapsed = @elapsed cache = with_logger(SimpleLogger(warning_buffer, Warn)) do
        update(initial; maxiter, tolerance,
               convergence = projector === :cycle ? :environment : :free_energy, verbose)
    end
    warning_text = String(take!(warning_buffer))
    isempty(warning_text) || print(stderr, warning_text)
    converged = verbose ? missing : !occursin("did not converge", warning_text)
    staggered_magnetisation = staggered_magnetisation_ctm(cache, plain, zinsert, signs)
    return cvm_freenergy(cache), staggered_magnetisation,
           marginal_inconsistency(cache), elapsed, converged
end

function write_results(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "method,chi,beta,temperature,n_sites,peps_bond_dimension,d_beta," *
                    "trotter_steps,staggered_field,lnZ,staggered_magnetisation," *
                    "reduced_free_energy_density,free_energy_density," *
                    "abs_error_lnZ,rel_error_lnZ,delta_from_previous_chi," *
                    "abs_error_staggered_magnetisation,rel_error_staggered_magnetisation," *
                    "delta_staggered_magnetisation_from_previous_chi," *
                    "marginal_inconsistency,solver_converged,wall_seconds,max_gate_error," *
                    "accumulated_log_scale,reference_method,reference_chi,reference_lnZ," *
                    "reference_staggered_magnetisation")
        for r in rows
            println(io, join((r.method, r.chi, r.beta, r.temperature, r.n_sites,
                              r.peps_bond_dimension, r.d_beta, r.trotter_steps,
                              r.staggered_field, r.lnZ, r.staggered_magnetisation,
                              r.reduced_free_energy_density, r.free_energy_density,
                              r.abs_error_lnZ, r.rel_error_lnZ, r.delta_from_previous_chi,
                              r.abs_error_staggered_magnetisation,
                              r.rel_error_staggered_magnetisation,
                              r.delta_staggered_magnetisation_from_previous_chi,
                              r.marginal_inconsistency, r.solver_converged, r.wall_seconds,
                              r.max_gate_error,
                              r.accumulated_log_scale, r.reference_method, r.reference_chi,
                              r.reference_lnZ, r.reference_staggered_magnetisation), ','))
        end
    end
    return path
end

function main()
    nx = envint("HEX_THERMAL_NX", 2)
    ny = envint("HEX_THERMAL_NY", 2)
    D = envint("HEX_THERMAL_D", 3)
    J = envfloat("HEX_THERMAL_J", 1.0)
    staggered_field = envfloat("HEX_THERMAL_STAGGERED_FIELD", 0.05)
    dβ = envfloat("HEX_THERMAL_DBETA", 0.02)
    nsteps = envint("HEX_THERMAL_STEPS", 5)
    cutoff = envfloat("HEX_THERMAL_CUTOFF", 1.0e-12)
    χs = sort(unique(envints("HEX_THERMAL_CHIS", [2, 4, 6, 8, 12, 16])))
    reference_χ = envint("HEX_THERMAL_REF_CHI", 32)
    maxiter = envint("HEX_THERMAL_CTM_MAXITER", 60)
    tolerance = envfloat("HEX_THERMAL_CTM_TOL", 1.0e-10)
    verbose = envbool("HEX_THERMAL_CTM_VERBOSE", false)
    output = get(ENV, "HEX_THERMAL_OUTPUT",
                 joinpath(@__DIR__, "data", "hexagonal_heisenberg_thermal_convergence.csv"))

    nx >= 1 && ny >= 1 || throw(ArgumentError("HEX_THERMAL_NX/NY must be positive"))
    D >= 1 || throw(ArgumentError("HEX_THERMAL_D must be positive"))
    dβ > 0 || throw(ArgumentError("HEX_THERMAL_DBETA must be positive"))
    nsteps >= 1 || throw(ArgumentError("HEX_THERMAL_STEPS must be positive"))
    !isempty(χs) && all(>(0), χs) || throw(ArgumentError(
        "HEX_THERMAL_CHIS must contain positive integers"))
    reference_χ >= maximum(χs) || throw(ArgumentError(
        "HEX_THERMAL_REF_CHI must be at least the largest scan χ"))

    β = 2nsteps * dβ
    temperature = inv(β)
    @printf("Finite open hexagonal AF Heisenberg thermal state: cells=%d×%d, D≤%d, β=%.4f (T=%.4f), h_s=%.4g, dβ=%.4f, steps=%d\n",
            nx, ny, D, β, temperature, staggered_field, dβ, nsteps)
    state, g, signs, max_gate_error, accumulated_log_scale = prepare_thermal_state(
        ; nx, ny, D, J, dβ, nsteps, cutoff, staggered_field)
    nsites = length(vertices(g))
    actual_D = maxvirtualdim(state)
    @printf("Prepared %d-site purification with final D=%d.\n", nsites, actual_D)
    tn, plain, zinsert = double_layer_network(state, g)

    reference_residual, reference_staggered_magnetisation, _, reference_time, _ =
        contract_bmps(tn, plain, zinsert, signs, reference_χ)
    reference_lnZ = accumulated_log_scale + reference_residual
    @printf("Reference bMPS χ=%d: lnZ=%.15g  -lnZ/N=%.15g  m_s=%.15g  %.2f s\n",
            reference_χ, reference_lnZ, -reference_lnZ / nsites,
            reference_staggered_magnetisation, reference_time)

    raw = NamedTuple[]
    methods = (("bMPS", nothing), ("svd-CTMRG", :cut), ("eig-CTMRG", :cycle))
    for (method, projector) in methods
        previous_lnZ = nothing
        previous_staggered_magnetisation = nothing
        for χ in χs
            residual, staggered_magnetisation, marginal, elapsed, solver_converged =
                isnothing(projector) ?
                contract_bmps(tn, plain, zinsert, signs, χ) :
                contract_ctm(tn, plain, zinsert, signs, χ, projector;
                             maxiter, tolerance, verbose)
            lnZ = accumulated_log_scale + residual
            delta = isnothing(previous_lnZ) ? NaN : abs(lnZ - previous_lnZ)
            delta_staggered_magnetisation = isnothing(previous_staggered_magnetisation) ? NaN :
                abs(staggered_magnetisation - previous_staggered_magnetisation)
            previous_lnZ = lnZ
            previous_staggered_magnetisation = staggered_magnetisation
            abs_error = abs(lnZ - reference_lnZ)
            rel_error = abs_error / max(abs(reference_lnZ), eps(Float64))
            abs_error_staggered_magnetisation = abs(
                staggered_magnetisation - reference_staggered_magnetisation)
            rel_error_staggered_magnetisation = abs_error_staggered_magnetisation /
                max(abs(reference_staggered_magnetisation), eps(Float64))
            push!(raw, (; method, chi = χ, beta = β, temperature, n_sites = nsites,
                         peps_bond_dimension = actual_D, d_beta = dβ,
                         trotter_steps = nsteps, staggered_field, lnZ,
                         staggered_magnetisation,
                         reduced_free_energy_density = -lnZ / nsites,
                         free_energy_density = -lnZ / (β * nsites),
                         abs_error_lnZ = abs_error, rel_error_lnZ = rel_error,
                         delta_from_previous_chi = delta,
                         abs_error_staggered_magnetisation,
                         rel_error_staggered_magnetisation,
                         delta_staggered_magnetisation_from_previous_chi =
                             delta_staggered_magnetisation,
                         marginal_inconsistency = marginal, solver_converged,
                         wall_seconds = elapsed,
                         max_gate_error, accumulated_log_scale, reference_method = "bMPS",
                         reference_chi = reference_χ, reference_lnZ,
                         reference_staggered_magnetisation))
            @printf("%-11s χ=%-3d lnZ=% .15e  |δlnZ|=%8.2e  m_s=% .12e  |δm_s|=%8.2e  marginal=%8.2e  converged=%-5s  %6.2f s\n",
                    method, χ, lnZ, abs_error, staggered_magnetisation,
                    abs_error_staggered_magnetisation, marginal,
                    string(solver_converged), elapsed)
        end
    end

    write_results(output, raw)
    println("Wrote $(abspath(output))")
    return raw
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
