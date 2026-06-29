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
using CUDA
using TensorNetworkQuantumSimulator: update, freenergy

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

function energy_doubleocc_bmps(ψ_bpc_row, ψ_bpc_col, U, t)
    ndubs = 0
    g = graph(ψ_bpc_row)
    for v in vertices(g)
        ndubs += only(expect(ψ_bpc_row, [(["NupNdn"], [v])]))
    end
    tot_kin_e = 0
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        if first(v1) == first(v2)
            tot_kin_e +=  expect(ψ_bpc_row, (["Cupdag", "Cup"], [src(e), dst(e)])) + expect(ψ_bpc_row, (["Cupdag", "Cup"], [dst(e), src(e)]))
            tot_kin_e +=  expect(ψ_bpc_row, (["Cdndag", "Cdn"], [src(e), dst(e)])) + expect(ψ_bpc_row, (["Cdndag", "Cdn"], [dst(e), src(e)]))
        else
            tot_kin_e +=  expect(ψ_bpc_col, (["Cupdag", "Cup"], [src(e), dst(e)])) + expect(ψ_bpc_col, (["Cupdag", "Cup"], [dst(e), src(e)]))
            tot_kin_e +=  expect(ψ_bpc_col, (["Cdndag", "Cdn"], [src(e), dst(e)])) + expect(ψ_bpc_col, (["Cdndag", "Cdn"], [dst(e), src(e)]))
        end
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
    dt = -0.01*im
    nsteps = 500
    g = named_hexagonal_lattice_graph(6,6; periodic = false)
    s = siteinds("spinful_fermion", g)

    println("Total number of sites is $(nv(g))")


    # --- the STATE: two spinful-fermion sites, z bonds; :A is up, :B is down ---
    # (both sites carry one fermion => equal parity, which the unit cell requires)
    ψ_bpc = deserialize("/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/OBCHex/States/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).ser")
    #ψ_bpc = update(ψ_bpc)

    Ntot_up = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])
    Ntot_dn = sum([expect(ψ_bpc, (["Ndn"], [v])) for v in vertices(g)])
    println("Total init spin up density is $(Ntot_up / length(vertices(g)))")
    println("Total init spin dn density is $(Ntot_dn / length(vertices(g)))")

    

    e, ndubs = energy_doubleocc(ψ_bpc, U, t_hop)
    e, ndubs = e / length(vertices(g)), ndubs / length(vertices(g))
    e = e - U/4

    println("BP NUMBERS:")
    println("Doublon density is $ndubs")
    println("Energy density is $e")

    ψ_cpu = network(ψ_bpc)
    ψ_gpu = CUDA.cu(ψ_cpu)

    #@time ψ_bpc_cpu = update(BeliefPropagationCache(ψ_cpu))
    #@time ψ_bpc_gpu = update(BeliefPropagationCache(ψ_gpu))

    #@show freenergy(ψ_bpc_gpu)
    #@show freenergy(ψ_bpc_cpu)


    Rs = [1,2,4,8,16,32]

    for R in Rs
        ψ_bmps_row = update(BoundaryMPSCache(ψ_gpu, R; partition_by = "row"))
        ψ_bmps_col = update(BoundaryMPSCache(ψ_gpu, R; partition_by = "col"))

        @show freenergy(ψ_bmps_row)

        e, ndubs = energy_doubleocc_bmps(ψ_bmps_row, ψ_bmps_col, U, t_hop)
        e, ndubs = e / length(vertices(g)), ndubs / length(vertices(g))
        e = e - U/4

        println("BMPS NUMBERS at R = $R:")
        println("Doublon density is $ndubs")
        println("Energy density is $e")
    end
end

U = 9.0
χ =4

#U = parse(Float64, ARGS[1])
#χ = parse(Int64, ARGS[2])
main(U, χ)
