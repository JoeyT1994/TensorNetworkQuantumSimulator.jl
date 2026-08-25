# Paper-matched 20x20 ferromagnetic random-bond Ising benchmark.
# Bonds take beta*J=0.88 or beta*J'=0.17 with equal probability; h=0.01.
# Every local observable uses an environment converged on the UNPERTURBED network.

include("ctm_rbim10_precision.jl")

function run_rbim20_jjprime()
    K, Kp, probability = 0.88, 0.17, 0.5
    L, h, site = 20, 0.01, (10, 10)
    chis = parseints("RBIM_CHIS", join(1:16, ','))
    seeds = parseints("RBIM_SEEDS", "401,402,403,404,405,406")
    reference_chi = parse(Int, get(ENV, "RBIM_REFERENCE_CHI", "24"))
    maxiter = parse(Int, get(ENV, "RBIM_MAXITER", "60"))
    tol = parse(Float64, get(ENV, "RBIM_TOL", "1e-10"))
    projectors = Symbol.(split(get(ENV, "RBIM_PROJECTORS", "cut,cycle,bmps"), ','))
    all(p -> p in (:cut, :cycle, :bmps), projectors) ||
        error("RBIM_PROJECTORS must contain cut, cycle, and/or bmps")

    @printf("20x20 paper RBIM: betaJ=%.2f betaJ'=%.2f p=%.1f h=%.3g site=%s\n",
            K, Kp, probability, h, string(site))
    @printf("chis=%s seeds=%s reference=cut(%d) maxiter=%d tolerance=%.1e\n",
            chis, seeds, reference_chi, maxiter, tol)
    println("seed,chi,projector,F_rel_error,m_abs_error,m_rel_error,marginal_inconsistency,seconds")

    for seed in seeds
        Random.seed!(seed)
        g = named_grid((L, L))
        couplings = Dictionary(collect(edges(g)),
                               [rand() < probability ? K : Kp for _ in edges(g)])
        # `rbim_setup` uses a=beta*J and beta*h. Passing beta=1 means `couplings` and h already
        # carry the dimensionless beta factors quoted in the paper.
        tn, mk, _ = rbim_setup(g, 1.0, h, couplings)
        tspin, tplain = mk(site; spin = true), mk(site)

        tref = @elapsed refcache = update(
            CTMEnvironmentCache(tn, reference_chi; projector = :cut);
            maxiter, tolerance = tol, convergence = :environment)
        Fref = cvm_freenergy(refcache)
        mref = local_magnetisation(refcache, tspin, tplain, site)
        @printf("# seed=%d F_ref=%.17g m_ref=%.17g ref_seconds=%.6f\n",
                seed, Fref, mref, tref)

        for χ in chis, method in projectors
            if method === :bmps
                elapsed = @elapsed F, m = bmps_freeenergy_and_local(
                    tn, tspin, tplain, site, χ)
                mi = NaN
            else
                elapsed = @elapsed cache = update(
                    CTMEnvironmentCache(tn, χ; projector = method, cycle_gapcut = 0.0);
                    maxiter, tolerance = tol, convergence = :environment)
                F = cvm_freenergy(cache)
                m = local_magnetisation(cache, tspin, tplain, site)
                mi = marginal_inconsistency(cache)
            end
            @printf("%d,%d,%s,%.17e,%.17e,%.17e,%.17e,%.6f\n",
                    seed, χ, method, abs(F - Fref) / abs(Fref), abs(m - mref),
                    abs(m - mref) / abs(mref), mi, elapsed)
            flush(stdout)
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_rbim20_jjprime()
