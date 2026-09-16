# Expects `model.jl` (the scratchpad problem builder: build_problem() with MODEL/L/G) in the same directory.
# Full update with finite-CTMRG environments on the L×L TFIM: imaginary-time Trotter gates applied
# with `full_update` in the ring of the norm network's CTM environments (`region_ring`), the
# environments re-converged (warm) after every colour layer of two-site gates (GROUP=1, default) or
# after every gate (GROUP=0). Baseline for the L-BFFS trajectory, timed like for like: every
# checkpoint records the environment-set wall clock, excluding process start and measurements.
#   START (state file; default the SU start tfim_LxL_D_g.jls), D, CHI, PROJ (cut), TOL, MAXIT (warm sweeps cap),
#   DTAU (0.02), NSTEPS per invocation, NFU (full-update sweeps, 10), TAG, MEAS (ring energy every MEAS steps)
include("model.jl")
using Dates, LinearAlgebra
using TensorNetworkQuantumSimulator: QuadraticForm, CTMEnvironmentCache, environments, region_ring, norm_factors, apply
BLAS.set_num_threads(1)
prob = build_problem(); g = prob.g; H = prob.H; nsites = prob.nsites
D = parse(Int, get(ENV, "D", "3")); χ = parse(Int, get(ENV, "CHI", "32")); proj = Symbol(get(ENV, "PROJ", "cut"))
tol = parse(Float64, get(ENV, "TOL", "1e-10")); maxit = parse(Int, get(ENV, "MAXIT", "30"))
dτ = parse(Float64, get(ENV, "DTAU", "0.02")); nsteps = parse(Int, get(ENV, "NSTEPS", "5")); nfu = parse(Int, get(ENV, "NFU", "10"))
group = get(ENV, "GROUP", "1") == "1"; meas = parse(Int, get(ENV, "MEAS", "1"))
gx = parse(Float64, get(ENV, "G", "3.0"))
tag = "$(prob.tag)_D$(D)_chi$(χ)_$(proj)_fu_dtau$(dτ)" * (group ? "" : "_pergate") * get(ENV, "TAG", "")
ckpt = "ckptfu_$tag.jls"; logf = "energiesfu_$tag.csv"

if isfile(ckpt)
    st = deserialize(ckpt); ψ = st.ψ; step0 = st.step; tenv = st.tenv; env0 = st.env
    println("resumed $ckpt after $step0 steps ($(round(tenv, digits = 1)) s of environment time)")
else
    L = isqrt(nsites); start = get(ENV, "START", "tfim_$(L)x$(L)_D$(D)_g$(gx).jls")
    st0 = deserialize(start); ψ = st0 isa TensorNetworkState ? st0 : st0.ψ; step0 = 0
    # a full-update checkpoint as START (dτ change): carry its environment time and warm environments
    tenv = (st0 isa TensorNetworkState || !haskey(st0, :tenv)) ? 0.0 : st0.tenv
    env0 = (st0 isa TensorNetworkState || !haskey(st0, :env)) ? nothing : st0.env
    for v in vertices(ψ)      # real Float64 tensors (the gates exp(θ X), exp(θ ZZ) are real)
        t = ψ[v]; is = collect(TI.inds(t))
        ψ[v] = TI.from_array(Array{Float64}(real.(TI.array(t, is...))), is...)
    end
    open(logf, "w") do io; println(io, "step,tau,E_ring,E_ring_per_site,t_env,timestamp"); end
end
sinds = siteinds(ψ)

# CTM cache of the norm network, seeded from `env` (warm) when given; returns (cache, seconds).
function norm_cache(ψ, env)
    c = CTMEnvironmentCache(QuadraticForm(ψ), χ; projector = proj)
    env === nothing || (c = TNQS._ctm_setenv(c, env))
    t = @elapsed c = update(c; maxiter = env === nothing ? 100 : maxit, tolerance = tol, convergence = :marginal)
    return c, t
end
# The 4C+4T ring of a vertex set contracted to ONE tensor (open legs: the set's outgoing ket and bra
# bonds, D^(2·nbonds) entries): `full_update` and the region energies search an optimal contraction
# sequence over their whole tensor list, which is fine for 7 tensors and not for 14.
ring_tensor(cache, vs) = TNQS._ctm_contract(region_ring(cache, vs), cache.options)
# Ring energy of the TFIM at the cache's environments: Σ_v −g⟨X_v⟩ + Σ_e −⟨Z Z⟩, each a region ratio.
function ring_energy(ψ, cache)
    E = 0.0
    for v in vertices(ψ)
        ring = vertex_ring(cache, v)
        num = TNQS._ctm_contract(vcat(ring, norm_factors(ψ, [v]; op_strings = _ -> "X")), cache.options)
        den = TNQS._ctm_contract(vcat(ring, norm_factors(ψ, [v])), cache.options)
        E -= gx * real(TNQS.scalar(num) / TNQS.scalar(den))
    end
    for e in edges(ψ)
        vs = [src(e), dst(e)]; ring = [ring_tensor(cache, vs)]
        num = TNQS._ctm_contract(vcat(ring, norm_factors(ψ, vs; op_strings = _ -> "Z")), cache.options)
        den = TNQS._ctm_contract(vcat(ring, norm_factors(ψ, vs)), cache.options)
        E -= real(TNQS.scalar(num) / TNQS.scalar(den))
    end
    return E
end

cache, t = norm_cache(ψ, env0); tenv += t
println("environments $(env0 === nothing ? "cold" : "warm") in $(round(t, digits = 1)) s"); flush(stdout)
if step0 == 0
    E = ring_energy(ψ, cache)
    println("step 0: E_ring = $E (per site $(E / nsites))")
    open(logf, "a") do io; println(io, "0,0.0,$E,$(E / nsites),$tenv,$(now())"); end
end
ec = edge_color(g, 4)
# second-order Trotter: exp(dτ g/2 ΣX) · Π_colours exp(dτ ZZ) · exp(dτ g/2 ΣX); Rx(θ) = exp(−iθX/2), Rzz(θ) = exp(−iθZZ/2)
rx = Dict(v => first(TNQS.totensor(("Rx", [v], im * gx * dτ), g, sinds)) for v in vertices(g))
rzz = Dict(e => first(TNQS.totensor(("Rzz", (src(e), dst(e)), 2im * dτ), g, sinds)) for e in edges(g))
realgate(t) = (is = collect(TI.inds(t)); TI.from_array(Array{Float64}(real.(TI.array(t, is...))), is...))
apply_onesite!(ψ) = (for v in vertices(ψ); ψ[v] = TNQS.normalize(TNQS.noprime(apply(realgate(rx[v]), ψ[v]))); end)
function apply_twosite!(ψ, cache, e)
    v1, v2 = src(e), dst(e)
    bond = only(TNQS.commoninds(ψ[v1], ψ[v2]))
    envs = [ring_tensor(cache, [v1, v2])]
    t1, t2 = TNQS.full_update(realgate(rzz[e]), ψ, [v1, v2]; envs, nfullupdatesweeps = nfu, maxdim = D, cutoff = nothing)
    newbond = only(TNQS.commoninds(t1, t2))
    TI.dim(newbond) == TI.dim(bond) || error("bond $(e) changed dimension $(TI.dim(bond)) → $(TI.dim(newbond))")
    ψ[v1] = TNQS.normalize(TNQS.replaceind(t1, newbond, bond)); ψ[v2] = TNQS.normalize(TNQS.replaceind(t2, newbond, bond))
    return ψ
end
function run!(ψ, cache, tenv)
  for step in (step0 + 1):(step0 + nsteps)
    tstep = time()
    apply_onesite!(ψ)
    cache, t = norm_cache(ψ, environments(cache)); tenv += t
    for colour in ec
        for e in colour
            apply_twosite!(ψ, cache, e)
            group || ((cache, t) = norm_cache(ψ, environments(cache)); tenv += t)
        end
        group && ((cache, t) = norm_cache(ψ, environments(cache)); tenv += t)
    end
    apply_onesite!(ψ)
    cache, t = norm_cache(ψ, environments(cache)); tenv += t
    if step % meas == 0
        tm = @elapsed E = ring_energy(ψ, cache)
        println("step $step (τ = $(round(step * dτ, digits = 3))): E_ring = $E (per site $(E / nsites)); env time $(round(tenv, digits = 1)) s total, step $(round(time() - tstep, digits = 1)) s (measurement $(round(tm, digits = 1)) s)")
        open(logf, "a") do io; println(io, "$step,$(step * dτ),$E,$(E / nsites),$tenv,$(now())"); end
    end
    flush(stdout)
    serialize(ckpt, (; ψ, step, tenv, env = environments(cache)))
  end
  return tenv
end
tenv = run!(ψ, cache, tenv)
println("invocation done: steps $step0 → $(step0 + nsteps); environment time $(round(tenv, digits = 1)) s")
