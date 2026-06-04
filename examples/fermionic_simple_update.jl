using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
Random.seed!(1234)

function main()
    ITensors.disable_warn_order()
    χ = 3
    g = named_hexagonal_lattice_graph(3,3)
    s = siteinds("spinful_fermion", g)
    ψ = fermionic_tensornetworkstate(ComplexF32, v-> isodd(sum(v)) ? "Up" : "Dn", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    dt = 0.1
    U = 5.0
    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14)
    single_site_gates = [fermionic_number_gate(dt, only(s[v]); coeff= - 0.5* im * U) for v in vertices(g)]
    two_site_gates =Vector{<:FermionicITensor}[]
    for es in ec
        push!(two_site_gates, [fermionic_hopping_gate(dt, only(s[src(e)]), only(s[dst(e)]); coeff = -t *im) for e in es])
    end

    @show expect(ψ_bpc, [(["Ndn"], [(1,1)])])
    nsteps = 5
    for i in 1:nsteps
        for single_site_gate in single_site_gates
            TensorNetworkQuantumSimulator.apply_gate!(single_site_gate,ψ_bpc;apply_kwargs)
        end

        for two_site_gate_group in two_site_gates
            for two_site_gate in two_site_gate_group
                TensorNetworkQuantumSimulator.apply_gate!(two_site_gate,ψ_bpc;apply_kwargs)
            end
        end

        for single_site_gate in single_site_gates
            TensorNetworkQuantumSimulator.apply_gate!(single_site_gate,ψ_bpc;apply_kwargs)
        end

        ψ_bpc = update(ψ_bpc)
    end

    @show expect(ψ_bpc, [(["Ndn"], [(1,1)])])
end

main()


