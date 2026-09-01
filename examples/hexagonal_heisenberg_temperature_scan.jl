"""
Fixed-χ temperature scan for the finite open honeycomb antiferromagnetic Heisenberg purification.

The state is evolved once and contracted at each requested inverse-temperature checkpoint. At every
checkpoint the same residual double-layer network is measured with bMPS, SVD-CTMRG (`:cut`), and
eig-CTMRG (`:cycle`) at a common χ. A separately converged high-χ bMPS contraction supplies the
reference, with a second high-χ bMPS point recorded as a convergence cross-check.

The staggered one-point function is weakly pinned because it vanishes exactly on a finite
spin-symmetric lattice. Defaults target a substantive but still workstation-sized run:

    HEX_TEMP_NX=4 HEX_TEMP_NY=4 HEX_TEMP_D=6 HEX_TEMP_CHI=12 \
    HEX_TEMP_BETAS=0.4,0.8,1.2,1.6,2.0 HEX_TEMP_STAGGERED_FIELD=0.05 \
    HEX_TEMP_DISORDER_STRENGTH=0.5 HEX_TEMP_DISORDER_SEED=271828 \
    julia --project=. examples/hexagonal_heisenberg_temperature_scan.jl

`HEX_TEMP_DISORDER_STRENGTH=W` draws one reproducible positive random-bond realization
`J_e = J(1 + W u_e)`, `u_e in [-1,1]`; the default `W=0` retains the clean model.
Set `HEX_TEMP_METHODS=svd-CTMRG,eig-CTMRG` for focused CTMRG reruns.
"""

include(joinpath(@__DIR__, "hexagonal_heisenbergmodel_thermalstate.jl"))
using Random: MersenneTwister, rand

envfloats(name, default) = parse.(Float64, split(get(ENV, name, join(default, ',')), ','))

function selected_temperature_methods()
    projectors = Dict("bMPS" => nothing, "svd-CTMRG" => :cut, "eig-CTMRG" => :cycle)
    names = strip.(split(get(ENV, "HEX_TEMP_METHODS", "bMPS,svd-CTMRG,eig-CTMRG"), ','))
    isempty(names) && throw(ArgumentError("HEX_TEMP_METHODS must select at least one method"))
    unknown = filter(name -> !haskey(projectors, name), names)
    isempty(unknown) || throw(ArgumentError(
        "unknown HEX_TEMP_METHODS entries $(join(unknown, ',')); choose bMPS, svd-CTMRG, eig-CTMRG"))
    return [(name, projectors[name]) for name in names]
end

function write_temperature_results(path, rows)
    mkpath(dirname(path))
    columns = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(columns, ','))
        for row in rows
            println(io, join((getproperty(row, column) for column in columns), ','))
        end
    end
    return path
end

function temperature_scan()
    nx = envint("HEX_TEMP_NX", 4)
    ny = envint("HEX_TEMP_NY", 4)
    D = envint("HEX_TEMP_D", 6)
    J = envfloat("HEX_TEMP_J", 1.0)
    disorder_strength = envfloat("HEX_TEMP_DISORDER_STRENGTH", 0.0)
    disorder_seed = envint("HEX_TEMP_DISORDER_SEED", 271828)
    staggered_field = envfloat("HEX_TEMP_STAGGERED_FIELD", 0.05)
    dβ = envfloat("HEX_TEMP_DBETA", 0.02)
    betas = sort(unique(envfloats("HEX_TEMP_BETAS", [0.4, 0.8, 1.2, 1.6, 2.0])))
    χ = envint("HEX_TEMP_CHI", 12)
    reference_χ = envint("HEX_TEMP_REF_CHI", 64)
    reference_check_χ = envint("HEX_TEMP_REF_CHECK_CHI", 48)
    cutoff = envfloat("HEX_TEMP_CUTOFF", 1.0e-12)
    maxiter = envint("HEX_TEMP_CTM_MAXITER", 60)
    tolerance = envfloat("HEX_TEMP_CTM_TOL", 1.0e-10)
    verbose = envbool("HEX_TEMP_CTM_VERBOSE", false)
    reference_only = envbool("HEX_TEMP_REFERENCE_ONLY", false)
    methods = selected_temperature_methods()
    output = get(ENV, "HEX_TEMP_OUTPUT",
                 joinpath(@__DIR__, "data", "hexagonal_heisenberg_temperature_scan.csv"))

    nx >= 1 && ny >= 1 || throw(ArgumentError("HEX_TEMP_NX/NY must be positive"))
    D >= 1 || throw(ArgumentError("HEX_TEMP_D must be positive"))
    0 <= disorder_strength < 1 || throw(ArgumentError(
        "HEX_TEMP_DISORDER_STRENGTH must satisfy 0 <= W < 1 so every bond remains antiferromagnetic"))
    dβ > 0 || throw(ArgumentError("HEX_TEMP_DBETA must be positive"))
    !isempty(betas) && all(>(0), betas) ||
        throw(ArgumentError("HEX_TEMP_BETAS must contain positive values"))
    χ > 0 || throw(ArgumentError("HEX_TEMP_CHI must be positive"))
    reference_χ >= χ || throw(ArgumentError("HEX_TEMP_REF_CHI must be at least HEX_TEMP_CHI"))
    reference_check_χ >= χ ||
        throw(ArgumentError("HEX_TEMP_REF_CHECK_CHI must be at least HEX_TEMP_CHI"))

    step_float = betas ./ (2dβ)
    steps = round.(Int, step_float)
    all(isapprox.(step_float, steps; atol = 1.0e-10, rtol = 0)) || throw(ArgumentError(
        "Every HEX_TEMP_BETAS value must be an integer multiple of 2*HEX_TEMP_DBETA=$(2dβ)"))
    checkpoint_by_step = Dict(zip(steps, betas))

    g = named_hexagonal_lattice_graph(nx, ny; periodic = false)
    signs = staggered_signs(g)
    rng = MersenneTwister(disorder_seed)
    couplings = Dict(e => J * (1 + disorder_strength * (2rand(rng) - 1)) for e in edges(g))
    coupling_min, coupling_max = extrema(values(couplings))
    s = siteinds("S=1/2", g; inds_per_site = 2)
    ψ = identity_tensornetworkstate(Float64, g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    layer = heisenberg_layer(s, g, couplings, dβ, staggered_field, signs)
    apply_kwargs = (; maxdim = D, cutoff, normalize_tensors = false)
    accumulated_log_scale = real(TNQS.freenergy(ψ_bpc))
    rescale!(ψ_bpc)
    max_gate_error = 0.0
    nsites = length(vertices(g))
    rows = NamedTuple[]

    @printf("Honeycomb temperature scan: cells=%d×%d, sites=%d, D≤%d, fixed χ=%d, h_s=%.4g, W=%.3g seed=%d J_range=[%.4f,%.4f], methods=%s, betas=%s\n",
            nx, ny, nsites, D, χ, staggered_field, disorder_strength, disorder_seed,
            coupling_min, coupling_max, join(first.(methods), ','), string(betas))

    for step in 1:maximum(steps)
        elapsed_evolution = @elapsed ψ_bpc, errors = apply_gates(
            layer, ψ_bpc; apply_kwargs, verbose = false)
        layer_error = isempty(errors) ? 0.0 : maximum(real.(errors))
        max_gate_error = max(max_gate_error, layer_error)
        accumulated_log_scale += real(TNQS.freenergy(ψ_bpc))
        rescale!(ψ_bpc)
        if step == 1 || step % 10 == 0 || haskey(checkpoint_by_step, step)
            @printf("evolution step %d/%d: β=%.3f  D=%d  max gate error=%.3e  %.2f s\n",
                    step, maximum(steps), 2step * dβ, maxvirtualdim(ψ_bpc),
                    max_gate_error, elapsed_evolution)
        end
        haskey(checkpoint_by_step, step) || continue

        β = checkpoint_by_step[step]
        temperature = inv(β)
        state = network(ψ_bpc)
        actual_D = maxvirtualdim(state)
        tn, plain, zinsert = double_layer_network(state, g)

        ref_residual, ref_ms, _, ref_seconds, _ =
            contract_bmps(tn, plain, zinsert, signs, reference_χ)
        check_residual, check_ms, _, check_seconds, _ =
            contract_bmps(tn, plain, zinsert, signs, reference_check_χ)
        reference_lnZ = accumulated_log_scale + ref_residual
        check_lnZ = accumulated_log_scale + check_residual
        @printf("checkpoint β=%.3f (T=%.4f): ref χ=%d lnZ=%.15g m_s=%.15g; χ=%d cross-check |δlnZ|=%.2e |δm_s|=%.2e\n",
                β, temperature, reference_χ, reference_lnZ, ref_ms, reference_check_χ,
                abs(check_lnZ - reference_lnZ), abs(check_ms - ref_ms))

        if reference_only
            push!(rows, (; beta = β, temperature, n_sites = nsites,
                         peps_bond_dimension = actual_D, d_beta = dβ,
                         trotter_steps = step, staggered_field, disorder_strength,
                         disorder_seed, coupling_min, coupling_max,
                         reference_method = "bMPS", reference_chi = reference_χ,
                         reference_lnZ,
                         reference_staggered_magnetisation = ref_ms,
                         reference_wall_seconds = ref_seconds,
                         reference_check_chi = reference_check_χ,
                         reference_check_lnZ = check_lnZ,
                         reference_check_staggered_magnetisation = check_ms,
                         reference_check_abs_delta_lnZ = abs(check_lnZ - reference_lnZ),
                         reference_check_abs_delta_staggered_magnetisation =
                             abs(check_ms - ref_ms),
                         reference_check_wall_seconds = check_seconds,
                         max_gate_error, accumulated_log_scale))
            continue
        end

        for (method, projector) in methods
            residual, ms, marginal, elapsed, converged = isnothing(projector) ?
                contract_bmps(tn, plain, zinsert, signs, χ) :
                contract_ctm(tn, plain, zinsert, signs, χ, projector;
                             maxiter, tolerance, verbose)
            lnZ = accumulated_log_scale + residual
            abs_error_lnZ = abs(lnZ - reference_lnZ)
            rel_error_lnZ = abs_error_lnZ / max(abs(reference_lnZ), eps(Float64))
            abs_error_staggered_magnetisation = abs(ms - ref_ms)
            rel_error_staggered_magnetisation = abs_error_staggered_magnetisation /
                max(abs(ref_ms), eps(Float64))
            push!(rows, (; method, chi = χ, beta = β, temperature,
                         n_sites = nsites, peps_bond_dimension = actual_D,
                         d_beta = dβ, trotter_steps = step, staggered_field,
                         disorder_strength, disorder_seed, coupling_min, coupling_max,
                         lnZ, staggered_magnetisation = ms,
                         abs_error_lnZ, rel_error_lnZ,
                         abs_error_staggered_magnetisation,
                         rel_error_staggered_magnetisation,
                         marginal_inconsistency = marginal,
                         solver_converged = converged, wall_seconds = elapsed,
                         max_gate_error, accumulated_log_scale,
                         reference_method = "bMPS", reference_chi = reference_χ,
                         reference_lnZ, reference_staggered_magnetisation = ref_ms,
                         reference_wall_seconds = ref_seconds,
                         reference_check_chi = reference_check_χ,
                         reference_check_abs_delta_lnZ = abs(check_lnZ - reference_lnZ),
                         reference_check_abs_delta_staggered_magnetisation =
                             abs(check_ms - ref_ms),
                         reference_check_wall_seconds = check_seconds))
            @printf("  %-11s χ=%-3d |δlnZ|=%8.2e  |δm_s|=%8.2e  marginal=%8.2e  converged=%-5s  %6.2f s\n",
                    method, χ, abs_error_lnZ, abs_error_staggered_magnetisation,
                    marginal, string(converged), elapsed)
            flush(stdout)
        end
    end

    write_temperature_results(output, rows)
    println("Wrote $(abspath(output))")
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    temperature_scan()
end
