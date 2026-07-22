using TensorNetworkQuantumSimulator
using CUDA
using LinearAlgebra
using TensorNetworkQuantumSimulator: update
using NPZ

# Exact U=0 (free-fermion) dynamics for the spinful quench, via the single-particle
# correlation matrix. h_{ab}=t on nearest-neighbour edges; up/down sectors are identical and
# uncorrelated, so <n_v↑>=<n_v↓>=C_vv and the double occupancy factorises: <n_v↑ n_v↓>=C_vv².
# Returns (nup_density[time], doccs[time][site]) over the requested `times`.
function exact_free_fermion_dynamics(g, t, times, v_measure)
    vs  = collect(vertices(g)); N = length(vs); Nmeasure = length(v_measure)
    idx = Dict(v => k for (k, v) in enumerate(vs))

    h = zeros(Float64, N, N)                       # single-particle hopping (one spin)
    for e in edges(g)
        a, b = idx[src(e)], idx[dst(e)]
        h[a, b] = t; h[b, a] = t
    end
    C0 = Diagonal([isodd(sum(v)) ? 1.0 : 0.0 for v in vs])
    A_vert_inds = findall(v -> isodd(sum(v)) && v ∈ v_measure, vs)
    B_vert_inds = findall(v -> iseven(sum(v)) && v ∈ v_measure, vs)

    cdw_order = Float64[]
    for T in times
        V = exp(-im * T * h)                       # e^{-i h T}
        C = conj(V) * C0 * transpose(V)            # <c†_i↑ c_j↑>(T)
        occ = real.(diag(C))
        push!(cdw_order, 2*(sum(occ[A_vert_inds]) - sum(occ[B_vert_inds]))/Nmeasure)
        
    end
    return cdw_order
end

function main(U, χ)
    g = named_hexagonal_lattice_graph(6,6)
    s = siteinds("spinful_fermion", g)
    ψ = fermionic_tensornetworkstate(v -> isodd(sum(v)) ? "UpDn" : "Emp", g, s)

    println("We have $(length(vertices(g))) sites")
    v_measure = collect(center(g))
    n_measure = length(v_measure)

    a_verts, b_verts = filter(v -> isodd(sum(v)), v_measure), filter(v -> iseven(sum(v)), v_measure)
    @assert length(a_verts) == length(b_verts)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    R = 2*χ
    dt = 0.01
    t = -1
    Δ = maximum([degree(g, v) for v in vertices(g)])
    ec = edge_color(g, Δ)

    ndts = 1000
    dt = 0.01; t = -1
    times = dt .* collect(1:ndts)
    cdw_order = exact_free_fermion_dynamics(g, t, times, v_measure)

    apply_kwargs= (; maxdim = χ, cutoff = 1e-14, normalize_tensors = true)
    single_site_gates = [("RInt", v, 0.5*U*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    n_tot_a, n_tot_b = sum([expect(ψ_bpc, (["N"], [v])) for v in a_verts]), sum([expect(ψ_bpc, (["N"], [v])) for v in b_verts])
    println("Init. CDW order is $((n_tot_a - n_tot_b) / (n_measure))")

    bp_cdw, bmps_cdw, exact_cdw = Float64[1.0], Float64[1.0], Float64[1.0]
    times = Float64[0.0]
    for i in 1:ndts

        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc = update(ψ_bpc)
        rescale!(ψ_bpc)

        if (i-1) % 10 == 0

            println("Maximum bond dimension is $(maxvirtualdim(ψ_bpc))")
            n_tot_a, n_tot_b = sum([expect(ψ_bpc, (["N"], [v])) for v in a_verts]), sum([expect(ψ_bpc, (["N"], [v])) for v in b_verts])
            println("BP CDW order is $((n_tot_a - n_tot_b) / (n_measure))")

            push!(bp_cdw, real((n_tot_a - n_tot_b) / (n_measure)))

            ψ_gpu = CUDA.cu(network(ψ_bpc))
            #ψ_cpu = network(ψ_bpc)
            ψ_bmps = update(BoundaryMPSCache(ψ_gpu, R))
            n_tot_a, n_tot_b = sum([expect(ψ_bmps, (["N"], [v])) for v in a_verts]), sum([expect(ψ_bmps, (["N"], [v])) for v in b_verts])

            println("BMPS CDW order is $((n_tot_a - n_tot_b) / (n_measure))")

            println("Exact CDW order is $(cdw_order[i])")

            push!(bmps_cdw, real((n_tot_a - n_tot_b) / (n_measure)))

            push!(exact_cdw, cdw_order[i])
            push!(times, i * dt)

            npzwrite("/mnt/home/jtindall/ceph/Data/Fermions/FermionDynamics/HexagoanlCDWQUenchU$(U)Chi$(χ).npz", times = times, bp_cdw = bp_cdw, bmps_cdw = bmps_cdw, exact_cdw = exact_cdw)
        end

    end
end

U, χ = parse(Float64, ARGS[1]), parse(Int64, ARGS[2])
#U, χ = 10.0, 16
main(U, χ)