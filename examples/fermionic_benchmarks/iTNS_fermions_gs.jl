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

function energy_doubleocc(ψ_bpc, U, t)

    ndubs = 0.5*(real(iTNS_expect(ψ_bpc, "NupNdn", :A)) + real(iTNS_expect(ψ_bpc, "NupNdn", :B)))
    up_hops = iTNS_expect(ψ_bpc, ["Cupdag", "Cup"], 1) + iTNS_expect(ψ_bpc, ["Cupdag", "Cup"], 2) + iTNS_expect(ψ_bpc, ["Cupdag", "Cup"], 3)
    dn_hops = iTNS_expect(ψ_bpc, ["Cdndag", "Cdn"], 1) + iTNS_expect(ψ_bpc, ["Cdndag", "Cdn"], 2) + iTNS_expect(ψ_bpc, ["Cdndag", "Cdn"], 3)
    hop_energy = 2*real(up_hops + dn_hops)*(-t)
    e = U*ndubs + hop_energy / 2

    return e, ndubs
end

function main(U, χ)
    z = 3            # coordination number (each site has z bonds)
    t_hop = 1.0      # hopping amplitude
    dt = -0.01*im
    nsteps = 2000

    # --- the STATE: two spinful-fermion sites, z bonds; :A is up, :B is down ---
    # (both sites carry one fermion => equal parity, which the unit cell requires)
    ψ = infinite_fermionic_tensornetworkstate(
        z; sitetype = "spinful_fermion", init = v -> v == :A ? "Up" : "Dn",
    )

    # --- wrap it in a BP cache and converge the messages ---
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc)

    apply_kwargs = (; maxdim = χ, cutoff = 1e-14, normalize_tensors = true)

    # one second-order Trotter step of exp(-i H dt)
    function trotter_step!(bpc)
        # half step of the on-site interaction on both sublattices
        for s in (:A, :B)
            bpc, _ = iTNS_apply_gate(bpc, ("RInt", 0.5*U * dt), s; apply_kwargs...)
            bpc, _ = iTNS_apply_gate(bpc, ("RN", -0.25*U * dt), s; apply_kwargs...)
        end
        for s in (:A, :B)

        end
        # full step of the hopping across every bond
        for k in 1:z
            bpc, _ = iTNS_apply_gate(bpc, ("RHop", -t_hop * dt), k; apply_kwargs...)
        end
        # the other half step of the interaction
        for s in (:A, :B)
            bpc, _ = iTNS_apply_gate(bpc, ("RInt", 0.5*U * dt), s; apply_kwargs...)
            bpc, _ = iTNS_apply_gate(bpc, ("RN", -0.25*U * dt), s; apply_kwargs...)
        end
        return bpc
    end

    e, ndubs = energy_doubleocc(ψ_bpc, U, t_hop)
    imaginary_times = Float64[]
    energies = Float64[e]
    double_occs = Float64[ndubs]

    for step in 0:nsteps
        if step > 0
            ψ_bpc = trotter_step!(ψ_bpc)
            ψ_bpc = update(ψ_bpc)
            rescale!(ψ_bpc)
        end
        t = step * dt
        e, ndubs = energy_doubleocc(ψ_bpc, U, t_hop)
        println("Time is $(abs(t))")
        println("Doublon density is $ndubs")
        println("Energy density is $e")
        push!(energies, e)
        push!(double_occs, ndubs)
        push!(imaginary_times, abs(step*dt))
        flush(stdout)
    end

    f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/iTNS/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).npz"
    npzwrite(f_str, energies = energies, imaginary_times = imaginary_times, double_occs = double_occs)

    serialize("/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/iTNS/States/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).ser", ψ_bpc)
end

# U = 3.1
# χ = 8

U = parse(Float64, ARGS[1])
χ = parse(Int64, ARGS[2])
main(U, χ)
