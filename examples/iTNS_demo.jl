# Demo: real-time quench of the transverse-field Ising model on an infinite
# z-coordinated (Bethe) lattice, using the two-site / z-bond InfiniteTensorNetworkState.
#
#   H = -J Σ_⟨ij⟩ Z_i Z_j  -  h Σ_i X_i
#
# Start from the all-up product state and watch the magnetisation ⟨Z⟩ relax.
# Run with:  julia --project=. examples/iTNS_demo.jl

using TensorNetworkQuantumSimulator
using Printf

function main()
    z = 3            # coordination number (each site has z bonds)
    J = 1.0
    h = 0.8
    dt = 0.05
    nsteps = 40
    maxdim = 8

    # --- the STATE: two sites, z bonds, all spins up ---
    ψ = infinite_tensornetworkstate(z; init = v -> "Up")

    # --- wrap it in a BP cache and converge the messages ---
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc)

    apply_kwargs = (; maxdim, cutoff = 1e-12)

    # one second-order Trotter step of exp(-i H dt)
    function trotter_step!(bpc)
        # half step of the field on both sublattices
        for s in (:A, :B)
            bpc, _ = iTNS_apply_gate(bpc, ("Rx", h * dt), s; apply_kwargs...)
        end
        # full step of the ZZ coupling across every bond
        for k in 1:z
            bpc, _ = iTNS_apply_gate(bpc, ("Rzz", -2 * J * dt), k; apply_kwargs...)
        end
        # the other half step of the field
        for s in (:A, :B)
            bpc, _ = iTNS_apply_gate(bpc, ("Rx", h * dt), s; apply_kwargs...)
        end
        return bpc
    end

    @printf("%6s  %10s  %10s  %12s\n", "t", "<Z>_A", "<X>_A", "<ZZ>_bond1")
    for step in 0:nsteps
        if step > 0
            ψ_bpc = trotter_step!(ψ_bpc)
            ψ_bpc = update(ψ_bpc)              # re-converge BP before measuring
        end
        t = step * dt
        mz = real(iTNS_expect(ψ_bpc, "Z", :A))
        mx = real(iTNS_expect(ψ_bpc, "X", :A))
        zz = real(iTNS_expect(ψ_bpc, "ZZ", 1))
        @printf("%6.2f  %10.6f  %10.6f  %12.6f\n", t, mz, mx, zz)
    end

    # the final cache still wraps a genuine state — recover it if you like
    ψ_final = InfiniteTensorNetworkState(network(ψ_bpc))
    return ψ_final
end

main()
