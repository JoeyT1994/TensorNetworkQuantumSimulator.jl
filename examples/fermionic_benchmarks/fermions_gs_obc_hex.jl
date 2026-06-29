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
using NamedGraphs
const NG = NamedGraphs

function energy_doubleocc(ψ_bpc, U, t)
    ndubs = 0
    g = graph(ψ_bpc)
    for v in vertices(g)
        ndubs += only(expect(ψ_bpc, [(["NupNdn"], [v])]))
    end
    tot_kin_e = 0
    for e in edges(g)
        tot_kin_e +=  expect(ψ_bpc, (["Cupdag", "Cup"], [src(e), dst(e)])) + expect(ψ_bpc, (["Cupdag", "Cup"], [dst(e), src(e)]))
        tot_kin_e +=  expect(ψ_bpc, (["Cdndag", "Cdn"], [src(e), dst(e)])) + expect(ψ_bpc, (["Cdndag", "Cdn"], [dst(e), src(e)]))
    end

    return -t  * tot_kin_e + U *ndubs, ndubs
end

function initial_state_f(g, v, δ)
    n = maximum(first.(collect(vertices(g))))
    return generate_evenly_doped_row(n, δ)[first(v)]
end

"""
    generate_evenly_doped_row(n::Int, delta::Float64)

Generates a deterministic vector of length `n` representing a configuration of physical sites:
- `"Up"`: Spin Up
- `"Dn"`: Spin Down
- `"Emp"`: Empty (Hole)

Constraints enforced:
- Holes ("Emp") are spaced as evenly as mathematically possible.
- Spins strictly alternate "Up", "Dn", "Up", "Dn"... across the remaining sites.
- n_up == n_dn (total spins must be even).
"""
function generate_evenly_doped_row(n::Int, delta::Float64)
    # 1. Calculate target number of holes and spins
    n_holes = round(Int, n * delta)
    n_spins = n - n_holes

    # 2. Enforce n_up == n_dn constraint (total spins must be even)
    if isodd(n_spins)
        # Shift to the nearest valid configuration
        if n_spins < n
            n_spins += 1
            n_holes -= 1
        else
            n_spins -= 1
            n_holes += 1
        end
    end

    # Initialize array with an empty string placeholder
    state = fill("", n)

    # 3. Distribute the holes evenly across the array
    if n_holes > 0
        spacing = n / n_holes
        for i in 1:n_holes
            # Calculate the centered position for each hole
            idx = round(Int, (i - 0.5) * spacing + 0.5)
            idx = clamp(idx, 1, n) # Prevent any floating point out-of-bounds
            state[idx] = "Emp"
        end
    end

    # 4. Fill the remaining spots with strictly alternating spins
    current_spin = "Up"
    for i in 1:n
        if state[i] == "" # If the site is not a hole
            state[i] = current_spin
            current_spin = current_spin == "Up" ? "Dn" : "Up" # Toggle spin
        end
    end

    # Output info block
    realized_delta = n_holes / n

    return state
end

function named_hexagonal_cylinder(ny::Int64)
    g = named_hexagonal_lattice_graph(ny, 3; periodic = false)

    column_lengths = length.([filter(v -> last(v) == i, collect(vertices(g))) for i in 1:4])

    g = NG.GraphsExtensions.add_vertex(g, (column_lengths[1] + 1, 1))
    g = NG.GraphsExtensions.add_vertex(g, (column_lengths[4] + 1, 4))
    g = NG.GraphsExtensions.add_edge(g, NamedEdge((column_lengths[1], 1) => (column_lengths[1] + 1, 1)))
    g = NG.GraphsExtensions.add_edge(g, NamedEdge((column_lengths[4], 4) => (column_lengths[4] + 1, 4)))

    column_1_vertices=  filter(v -> last(v) == 1, collect(vertices(g)))

    for v in column_1_vertices
       if isempty(filter(v -> last(v) == 2, neighbors(g,v)))
            g = NG.GraphsExtensions.add_edge(g, NamedEdge(v => (first(v), 4)))
       end
    end

    return g
end


function main(U, χ)
    # coordination number (each site has z bonds)
    t_hop = 1.0      # hopping amplitude
    dt = -0.05*im
    nsteps = 500
    g = named_hexagonal_lattice_graph(6,6; periodic = false)
    s = siteinds("spinful_fermion", g)

    println("Total number of sites is $(nv(g))")


    # --- the STATE: two spinful-fermion sites, z bonds; :A is up, :B is down ---
    # (both sites carry one fermion => equal parity, which the unit cell requires)
    ψ = fermionic_tensornetworkstate(Float64,v -> isodd(sum(v)) ? "Up" : "Dn", g, s)
    
    # --- wrap it in a BP cache and converge the messages ---
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc)

    Ntot_up = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])
    Ntot_dn = sum([expect(ψ_bpc, (["Ndn"], [v])) for v in vertices(g)])
    println("Total init spin up density is $(Ntot_up / length(vertices(g)))")
    println("Total init spin dn density is $(Ntot_dn / length(vertices(g)))")

    μ = U / 2

    apply_kwargs = (; maxdim = χ, cutoff = 1e-14, normalize_tensors = true)
    single_site_gates = [("RInt", v, U * dt) for v in vertices(g)]
    single_site_gates = [single_site_gates; [("RN", v, -μ * dt) for v in vertices(g)]]
    ec = edge_color(g, 3)
    two_site_gates = []
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], -t_hop * dt) for e in es])
    end
    e, ndubs = energy_doubleocc(ψ_bpc, U, t_hop)
    imaginary_times = Float64[]
    energies = ComplexF64[e]
    double_occs = ComplexF64[ndubs]
    gates = [single_site_gates; two_site_gates]

    for step in 1:nsteps
        ψ_bpc, _ = apply_gates(gates, ψ_bpc; apply_kwargs, update_cache =false)

        if step % 2 == 0
            ψ_bpc = update(ψ_bpc)
            rescale!(ψ_bpc)
            t = step * dt
            e, ndubs = energy_doubleocc(ψ_bpc, U, t_hop)
            println("Time is $(abs(t))")
            println("Doublon density is $(ndubs / length(vertices(g)))")
            println("Energy density is $(e/ length(vertices(g)))")
            Ntot = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])
            println("Total spin up is $(Ntot / length(vertices(g)))")
            push!(energies, e)
            push!(double_occs, ndubs)
            push!(imaginary_times, abs(step*dt))
            flush(stdout)
        end

        if step % 100 == 0
            serialize("/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/OBCHex/States/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).ser", ψ_bpc)
        end
    end

    f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/OBCHex/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).npz"
    npzwrite(f_str, energies = energies, imaginary_times = imaginary_times, double_occs = double_occs)

    serialize("/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/OBCHex/States/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).ser", ψ_bpc)
end

U = 9.0
χ =4

#U = parse(Float64, ARGS[1])
#χ = parse(Int64, ARGS[2])
main(U, χ)
