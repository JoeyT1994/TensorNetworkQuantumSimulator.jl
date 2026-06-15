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
using Printf

function main()
    z = 3            # coordination number (each site has z bonds)
    t_hop = 1.0      # hopping amplitude
    U = 4.0          # on-site Hubbard interaction (set to 0.0 for free fermions)
    dt = 0.05
    nsteps = 40
    maxdim = 8

    # --- the STATE: two spinful-fermion sites, z bonds; :A is up, :B is down ---
    # (both sites carry one fermion => equal parity, which the unit cell requires)
    ψ = infinite_fermionic_tensornetworkstate(
        z; sitetype = "spinful_fermion", init = v -> v == :A ? "Up" : "Dn",
    )

    # --- wrap it in a BP cache and converge the messages ---
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc)

    apply_kwargs = (; maxdim, cutoff = 1e-12)

    # one second-order Trotter step of exp(-i H dt)
    function trotter_step!(bpc)
        # half step of the on-site interaction on both sublattices
        for s in (:A, :B)
            bpc, _ = iTNS_apply_gate(bpc, ("RInt", U * dt), s; apply_kwargs...)
        end
        # full step of the hopping across every bond
        for k in 1:z
            bpc, _ = iTNS_apply_gate(bpc, ("RHop", -2 * t_hop * dt), k; apply_kwargs...)
        end
        # the other half step of the interaction
        for s in (:A, :B)
            bpc, _ = iTNS_apply_gate(bpc, ("RInt", U * dt), s; apply_kwargs...)
        end
        return bpc
    end

    @printf("%6s  %10s  %10s  %12s  %12s\n", "t", "<Nup>_A", "<Ndn>_A", "<NupNdn>_A", "<N>_total")
    for step in 0:nsteps
        if step > 0
            ψ_bpc = trotter_step!(ψ_bpc)
            ψ_bpc = update(ψ_bpc)              # re-converge BP before measuring
        end
        t = step * dt
        nupA = real(iTNS_expect(ψ_bpc, "Nup", :A))
        ndnA = real(iTNS_expect(ψ_bpc, "Ndn", :A))
        dblA = real(iTNS_expect(ψ_bpc, "NupNdn", :A))
        nupB = real(iTNS_expect(ψ_bpc, "Nup", :B))
        ndnB = real(iTNS_expect(ψ_bpc, "Ndn", :B))
        ntot = nupA + ndnA + nupB + ndnB
        @printf("%6.2f  %10.6f  %10.6f  %12.6f  %12.6f\n", t, nupA, ndnA, dblA, ntot)
    end

    # the final cache still wraps a genuine state — recover it if you like
    ψ_final = InfiniteTensorNetworkState(network(ψ_bpc))
    return ψ_final
end

main()
