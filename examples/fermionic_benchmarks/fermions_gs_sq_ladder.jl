# Demo: real-time quench of the Hubbard model on an infinite z-coordinated
# (Bethe) lattice, using the two-site / z-bond fermionic InfiniteTensorNetworkState.
#
#   H = -t Σ_⟨ij⟩,σ (c†_{iσ} c_{jσ} + h.c.)  +  U Σ_i n↑_i n↓_i
#
# Start from a charge-density / spin-ordered product state (sublattice :A fully
# spin-up, :B fully spin-down — equal parity, as the unit cell requires) and
# watch the particles hop and the up/down densities relax.  Setting U = 0 gives
# the exactly-solvable free-fermion hopping quench.
#
# Run with:  julia --project=. examples/iTNS_fermions_demo.jl

using TensorNetworkQuantumSimulator
using NPZ
using Serialization
using TensorNetworkQuantumSimulator: expect
using CUDA

function energy_doubleocc(ψ_bpc, U, t_perp, t_parr)
    ndubs = 0
    g = graph(ψ_bpc)
    for v in vertices(g)
        ndubs += only(expect(ψ_bpc, [(["NupNdn"], [v])]))
    end
    tot_kin_e = 0
    for e in edges(g)
        t_hop = src(e)[1] == dst(e)[1] ? t_perp : t_parr
        tot_kin_e +=  t_hop*(expect(ψ_bpc, (["Cupdag", "Cup"], [src(e), dst(e)])) + expect(ψ_bpc, (["Cupdag", "Cup"], [dst(e), src(e)])))
        tot_kin_e +=  t_hop*(expect(ψ_bpc, (["Cdndag", "Cdn"], [src(e), dst(e)])) + expect(ψ_bpc, (["Cdndag", "Cdn"], [dst(e), src(e)])))
    end

    return -tot_kin_e + U *ndubs, ndubs
end

function energy_doubleocc_bmps(ψ_bpc_row, ψ_bpc_col, U, g, t_perp, t_parr)
    ndubs = 0
    for v in vertices(g)
        ndubs += only(expect(ψ_bpc_row, [(["NupNdn"], [v])]))
    end
    tot_kin_e = 0
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        t_hop = src(e)[1] == dst(e)[1] ? t_perp : t_parr
        if first(v1) == first(v2)
            tot_kin_e +=  t_hop*(expect(ψ_bpc_row, (["Cupdag", "Cup"], [src(e), dst(e)])) + expect(ψ_bpc_row, (["Cupdag", "Cup"], [dst(e), src(e)])))
            tot_kin_e +=  t_hop*(expect(ψ_bpc_row, (["Cdndag", "Cdn"], [src(e), dst(e)])) + expect(ψ_bpc_row, (["Cdndag", "Cdn"], [dst(e), src(e)])))
        else
            tot_kin_e +=  t_hop*(expect(ψ_bpc_col, (["Cupdag", "Cup"], [src(e), dst(e)])) + expect(ψ_bpc_col, (["Cupdag", "Cup"], [dst(e), src(e)])))
            tot_kin_e +=  t_hop*(expect(ψ_bpc_col, (["Cdndag", "Cdn"], [src(e), dst(e)])) + expect(ψ_bpc_col, (["Cdndag", "Cdn"], [dst(e), src(e)])))
        end
    end

    return -tot_kin_e + U *ndubs, ndubs
end

function initial_state_f(v)
    (v == (4,4) || v == (4,5)) && return "Emp"
    isodd(sum(v)) && return "Up"
    return "Dn"
end


function main(U, χ)
    # coordination number (each site has z bonds)
    t_parr, t_perp= 1.0, 0.5
    dt = -0.01*im
    nsteps = 1000
    n =10
    g = named_grid((n,2); periodic = false)
    s = TensorNetworkQuantumSimulator.siteinds("spinful_fermion", g)

    # --- the STATE: two spinful-fermion sites, z bonds; :A is up, :B is down ---
    # (both sites carry one fermion => equal parity, which the unit cell requires)
    δ = 0
    n = 1 +δ
    Ntot = (length(vertices(g))*n)
    Nup, Ndn = Ntot ÷ 2, Ntot ÷ 2
    ψ = fermionic_tensornetworkstate(v -> initial_state_f(v), g, s)

    e_offset = (U/2)*length(vertices(g))

    # --- wrap it in a BP cache and converge the messages ---
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc)

    μ = U / 2

    apply_kwargs = (; maxdim = χ, cutoff = 1e-14, normalize_tensors = true)
    single_site_gates = [("RInt", v, 0.5*U * dt) for v in vertices(g)]
    single_site_gates = [single_site_gates; [("RN", v, -0.5*μ * dt) for v in vertices(g)]]
    ec = edge_color(g, 3)
    two_site_gates = []
    for es in ec
        for e in es
            t_hop = src(e)[1] == dst(e)[1] ? t_perp : t_parr
            push!(two_site_gates, ("RHop", [src(e), dst(e)], -t_hop * dt))
        end
    end
    e, ndubs = energy_doubleocc(ψ_bpc, U, t_perp, t_parr)
    println("Initial Energy is $(e - e_offset)")
    imaginary_times = Float64[]
    energies = Float64[e]
    double_occs = Float64[ndubs]
    gates = [single_site_gates; two_site_gates; single_site_gates]

    for step in 1:nsteps
        ψ_bpc, _ = apply_gates(gates, ψ_bpc; apply_kwargs, update_cache = false)

        if step % 4 == 0
            ψ_bpc = update(ψ_bpc)
            rescale!(ψ_bpc)
            t = step * dt
            e, ndubs = energy_doubleocc(ψ_bpc, U, t_perp, t_parr)
            println("Time is $(abs(t))")
            println("Doublon density is $(ndubs / length(vertices(g)))")
            println("Energy is $(e - e_offset)")
            Ntot = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])
            println("Total spin up is $(Ntot / length(vertices(g)))")
            push!(energies, e)
            push!(double_occs, ndubs)
            push!(imaginary_times, abs(step*dt))

            if step % 1000 == 0 || step == 4
                R = 3*χ
                ψ = CUDA.cu(network(ψ_bpc))
                ψ_bmps_row = update(BoundaryMPSCache(ψ, R; partition_by = "row"))
                ψ_bmps_col = update(BoundaryMPSCache(ψ, R; partition_by = "col"))
                e_bmps, ndubs_bmps = energy_doubleocc_bmps(ψ_bmps_row, ψ_bmps_col, U, g, t_perp, t_parr)
                println("BMPS measured energy is $(e_bmps - e_offset)")
            end

            flush(stdout)
        end
    end
end

U = 8.0
χ =16
main(U, χ)
