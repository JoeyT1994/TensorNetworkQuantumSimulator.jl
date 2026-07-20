using TensorNetworkQuantumSimulator

using LinearAlgebra

# Exact U=0 (free-fermion) dynamics for the spinful quench, via the single-particle
# correlation matrix. h_{ab}=t on nearest-neighbour edges; up/down sectors are identical and
# uncorrelated, so <n_v↑>=<n_v↓>=C_vv and the double occupancy factorises: <n_v↑ n_v↓>=C_vv².
# Returns (nup_density[time], doccs[time][site]) over the requested `times`.
function exact_free_fermion_dynamics(g, occ_sites, t, times)
    vs  = collect(vertices(g)); N = length(vs)
    idx = Dict(v => k for (k, v) in enumerate(vs))

    h = zeros(Float64, N, N)                       # single-particle hopping (one spin)
    for e in edges(g)
        a, b = idx[src(e)], idx[dst(e)]
        h[a, b] = t; h[b, a] = t
    end
    C0 = Diagonal([v in occ_sites ? 1.0 : 0.0 for v in vs])   # occupied on the UpDn sites

    nup_density = Float64[]; doccs = Vector{Float64}[]
    for T in times
        V = exp(-im * T * h)                       # e^{-i h T}
        C = conj(V) * C0 * transpose(V)            # <c†_i↑ c_j↑>(T)
        occ = real.(diag(C))
        push!(nup_density, sum(occ) / N)
        push!(doccs, [occ[idx[v]]^2 for v in occ_sites])
    end
    return nup_density, doccs
end

function main()
    g = named_grid((11,11))
    s = siteinds("spinful_fermion", g)
    occ_sites = [(6,6), (6,5),(6,7),(5,6),(7,6)]
    ψ = fermionic_tensornetworkstate(v -> v ∈ occ_sites ? "UpDn" : "Emp", g, s)

    ψ_bpc = update(BeliefPropagationCache(ψ))
    χ = 6
    U = 0
    dt = 0.01
    t = -1
    ec = edge_color(g, 4)

    ndts = 100
    dt = 0.01; t = -1
    times = dt .* collect(1:ndts)
    nup_exact, doccs_exact = exact_free_fermion_dynamics(g, occ_sites, t, times)

    apply_kwargs= (; maxdim = χ, cutoff = 1e-14, normalize_tensors = true)
    single_site_gates = [("RInt", v, 0.5*U*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nup_tot = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])
    println("Init. Nup density is $(nup_tot / length(vertices(g)))")

    for i in 1:ndts

        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc = update(ψ_bpc)
        rescale!(ψ_bpc)

        if (i-1) % 10 == 0
            nup_tot = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])
            println("Total Nup density is $(nup_tot / length(vertices(g)))")

            println("Maximum bond dimension is $(maxvirtualdim(ψ_bpc))")

            doccs_bp = real.(expect(ψ_bpc, [(["NupNdn"], [v]) for v in occ_sites]))
            doccs_bmps = real.(expect(network(ψ_bpc), [(["NupNdn"], [v]) for v in occ_sites]; alg = "boundarymps", mps_bond_dimension = 2*χ))

            @show abs(doccs_bp[1] - doccs_exact[i][1])
            @show abs(doccs_bmps[1] - doccs_exact[i][1])
        end

    end
end

main()