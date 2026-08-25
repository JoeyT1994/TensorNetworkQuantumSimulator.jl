# Precision benchmark for finite CTMRG on the 10x10 random-bond Ising model used in
# `ctm_vs_bmps_rbim.jl`. Unlike that disorder-averaged comparison, this script records both the
# CVM free energy and a local magnetisation, together with convergence diagnostics.
#
# Run:
#   julia --project=. --startup-file=no examples/ctm_rbim10_precision.jl
# Optional environment variables:
#   RBIM_CHIS=8,12,16  RBIM_SEEDS=101,102  RBIM_MAXITER=60  RBIM_TOL=1e-12
#   RBIM_REFERENCE=bmps  RBIM_REFERENCE_CHI=32  RBIM_CYCLE_CONVERGENCE=free_energy
#   RBIM_CYCLE_GAPCUT=1e-4
#   RBIM_CYCLE_SUBSPACE=false  RBIM_CYCLE_ITERS=20  RBIM_CYCLE_WARMSTART=true
#   RBIM_PROJECTORS=cut,cycle,bmps
# To skip reference contraction during solver diagnostics, provide both cached values:
#   RBIM_REFERENCE_F=161.08999787608553  RBIM_REFERENCE_M=0.6287360485837528

using TensorNetworkQuantumSimulator, ITensors, Printf, Random
using Dictionaries: Dictionary, set!
using NamedGraphs.GraphsExtensions: incident_edges

const T = TensorNetworkQuantumSimulator
const BMPS_KW = (; message_update_alg = T.Algorithm("zipup"; cutoff = 0.0))

function rbim_setup(g, β, h, Js)
    links = Dictionary(edges(g), [Index(2, "e$(src(e))_$(dst(e))") for e in edges(g)])
    links = merge(links, Dictionary(reverse.(edges(g)), [links[e] for e in edges(g)]))
    sW = Dictionary()
    for e in edges(g)
        a = β * Js[e]
        a = a < 0 ? Complex(a) : a
        λ1, λ2 = cosh(a), sinh(a)
        α, ϕ = 0.5 * (sqrt(λ1) + sqrt(λ2)), 0.5 * (sqrt(λ1) - sqrt(λ2))
        M = sqrt(2) * [α ϕ; ϕ α]
        set!(sW, e, M)
        set!(sW, reverse(e), M)
    end
    function mk(v; spin = false)
        es = collect(incident_edges(g, v; dir = :in))
        w = [exp(β * h * s) * (spin ? s : 1.0) for s in (+1.0, -1.0)]
        A = zeros(ComplexF64, ntuple(_ -> 2, length(es)))
        for si in 1:2, idx in CartesianIndices(size(A))
            A[idx] += w[si] * prod(sW[es[k]][si, idx[k]] for k in eachindex(es))
        end
        return ITensor(A, [links[e] for e in es])
    end
    vs = collect(vertices(g))
    tn = T.TensorNetwork(Dictionary(vs, [mk(v) for v in vs]), g)
    tnspin(v) = T.TensorNetwork(Dictionary(vs, [u == v ? mk(u; spin = true) : mk(u) for u in vs]), g)
    return tn, mk, tnspin
end

function local_magnetisation(cache, tspin, tplain, v)
    env = vertex_ring(cache, v)
    num = scalar(T._ctm_contract(ITensor[tspin; env], T.options(cache)))
    den = scalar(T._ctm_contract(ITensor[tplain; env], T.options(cache)))
    return real(num / den)
end

function bmps_freeenergy_and_local(tn, tspin, tplain, v, χ)
    cache = T.BoundaryMPSCache(tn, χ; partition_by = "row", gauge_state = false)
    cache = update(cache; BMPS_KW...)
    F = log(abs(partitionfunction(cache)))
    # Keep the environment obtained from the UNPERTURBED network. `update_partitions` only brings
    # the messages inside the observable's row into contraction form; it does not solve a perturbed
    # network. Numerator and denominator then differ solely at the local site tensor.
    local_cache = T.update_partitions(cache, [v])
    env = T.incoming_messages(local_cache, v)
    num = scalar(contract(ITensor[tspin; env]))
    den = scalar(contract(ITensor[tplain; env]))
    return F, real(num / den)
end

parseints(name, default) = parse.(Int, split(get(ENV, name, default), ','))

function main()
    p = 0.109
    β = 0.5 * log((1 - p) / p)
    L, h, site = 10, 0.01, (4, 4)
    chis = parseints("RBIM_CHIS", "8,12,16")
    seeds = parseints("RBIM_SEEDS", "101")
    maxiter = parse(Int, get(ENV, "RBIM_MAXITER", "60"))
    tol = parse(Float64, get(ENV, "RBIM_TOL", "1e-12"))
    reference = Symbol(get(ENV, "RBIM_REFERENCE", "exact"))
    reference_chi = parse(Int, get(ENV, "RBIM_REFERENCE_CHI", "32"))
    cycle_convergence = Symbol(get(ENV, "RBIM_CYCLE_CONVERGENCE", "environment"))
    cycle_gapcut = parse(Float64, get(ENV, "RBIM_CYCLE_GAPCUT", "0"))
    cycle_subspace = parse(Bool, get(ENV, "RBIM_CYCLE_SUBSPACE", "false"))
    cycle_iters = parse(Int, get(ENV, "RBIM_CYCLE_ITERS", "20"))
    cycle_warmstart = parse(Bool, get(ENV, "RBIM_CYCLE_WARMSTART", "true"))
    projectors = Symbol.(split(get(ENV, "RBIM_PROJECTORS", "cut,cycle"), ','))
    all(p -> p in (:cut, :cycle, :bmps), projectors) ||
        error("RBIM_PROJECTORS must contain cut, cycle, and/or bmps")
    reference in (:exact, :bmps) || error("RBIM_REFERENCE must be exact or bmps")
    cycle_convergence in (:free_energy, :environment, :worst_region) ||
        error("RBIM_CYCLE_CONVERGENCE must be free_energy, environment, or worst_region")

    @printf("10x10 RBIM precision benchmark: p=%.3f beta=%.12f h=%.3g site=%s\n",
            p, β, h, string(site))
    @printf("chis=%s seeds=%s maxiter=%d tolerance=%.1e reference=%s(%d) cycle_convergence=%s cycle_gapcut=%.1e cycle_subspace=%s(%d)\n",
            chis, seeds, maxiter, tol, reference, reference_chi, cycle_convergence, cycle_gapcut,
            cycle_subspace, cycle_iters)
    println("seed,chi,projector,F_error,m_abs_error,m_rel_error,marginal_inconsistency,seconds")

    for seed in seeds
        Random.seed!(seed)
        g = named_grid((L, L))
        Js = Dictionary(collect(edges(g)), [rand() < p ? -1.0 : 1.0 for _ in edges(g)])
        tn, mk, tnspin = rbim_setup(g, β, h, Js)
        spin_tn = tnspin(site)

        cached_F = get(ENV, "RBIM_REFERENCE_F", nothing)
        cached_m = get(ENV, "RBIM_REFERENCE_M", nothing)
        if !isnothing(cached_F) && !isnothing(cached_m)
            Fexact = parse(Float64, cached_F)
            mexact = parse(Float64, cached_m)
        elseif reference === :exact
            Z = contract(tn; alg = "exact")
            Zspin = contract(spin_tn; alg = "exact")
            Fexact = log(abs(Z))
            mexact = real(Zspin / Z)
        else
            Fexact, mexact = bmps_freeenergy_and_local(
                tn, mk(site; spin = true), mk(site), site, reference_chi)
        end
        @printf("# seed=%d F_exact=%.17g m_exact=%.17g\n", seed, Fexact, mexact)

        for χ in chis, projector in projectors
            if projector === :bmps
                elapsed = @elapsed F, m = bmps_freeenergy_and_local(
                    tn, mk(site; spin = true), mk(site), site, χ)
                mi = NaN
            else
                elapsed = @elapsed cache = update(
                    CTMEnvironmentCache(tn, χ; projector,
                                        cycle_gapcut = projector === :cycle ? cycle_gapcut : 1e-4,
                                        cycle_subspace = projector === :cycle && cycle_subspace,
                                        cycle_iters, cycle_warmstart);
                    maxiter,
                    tolerance = tol,
                    convergence = projector === :cycle ? cycle_convergence : :free_energy,
                )
                F = cvm_freenergy(cache)
                m = local_magnetisation(cache, mk(site; spin = true), mk(site), site)
                mi = marginal_inconsistency(cache)
            end
            @printf("%d,%d,%s,%.17e,%.17e,%.17e,%.17e,%.6f\n",
                    seed, χ, projector, abs(F - Fexact), abs(m - mexact),
                    abs(m - mexact) / abs(mexact), mi, elapsed)
            flush(stdout)
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
