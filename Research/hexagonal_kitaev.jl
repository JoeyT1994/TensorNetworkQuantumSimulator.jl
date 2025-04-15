using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph

function hexagonal_kitaev_layer(Jx::Float64, Jy::Float64, Jz::Float64, δβ::Float64, ec)
    layer = []
    for (i, colored_edges) in enumerate(ec)
        if i == 1
            append!(layer, ("Rxx", pair, -2*Jx*δβ * im) for pair in colored_edges)
        elseif i == 2
            append!(layer, ("Ryy", pair, -2*Jy*δβ * im) for pair in colored_edges)
        elseif i == 3
            append!(layer, ("Rzz", pair, -2*Jz*δβ * im) for pair in colored_edges)
        end
    end
    return layer
end

function hexagonal_kitaev_observables(Jx::Float64, Jy::Float64, Jz::Float64, ec)
    xx_observables = [("XX", pair, Jx) for pair in ec[1]]
    yy_observables = [("YY", pair, Jy) for pair in ec[2]]
    zz_observables = [("ZZ", pair, Jz) for pair in ec[3]]
    return xx_observables, yy_observables, zz_observables
end

#Project spins on sites v1 and v2 to v1_val (1 = up, 2 = down) and v2_val
function project(ψIψ, v1, v2, v1_val::Int64 = 1, v2_val::Int64=1)
    ψIψ = copy(ψIψ)
    s1 = only(inds(only(TN.factors(ψIψ, [(v1, "operator")])); plev = 0))
    s2 = only(inds(only(TN.factors(ψIψ, [(v2, "operator")])); plev = 0))
    ψIψ = TN.update_factor(ψIψ, (v1, "operator"), onehot(s1 => v1_val) * dag(onehot(s1' => v1_val)))
    ψIψ = TN.update_factor(ψIψ, (v2, "operator"), onehot(s2 => v2_val) * dag(onehot(s2' => v2_val)))

    return ψIψ
end 

function main()

    f = load("Research/nx3ny3nz8Chi16DisorderNo10AnnealingTime7.jld2")
    ψ = f["Wavefunction"]
    ψIψ = build_bp_cache(ψ)

    ψ, ψIψ = normalize(ψ, ψIψ)

    v1, v2 = 11, 17

    bp_update_kwargs = get_global_bp_update_kwargs()

    obs = ("ZZ", [v1, v2])
    ψOψ = project(ψIψ, v1, v2, 1, 1)
    ψOψ = TN.update(ψOψ; bp_update_kwargs...)
    pupup = scalar(ψOψ) * scalar(normalize(ψOψ; update_cache = false); alg = "loopcorrections", max_configuration_size = 16)

    ψOψ = project(ψIψ, v1, v2, 1, 2)
    ψOψ = TN.update(ψOψ; bp_update_kwargs...)
    pupdown = scalar(ψOψ) * scalar(normalize(ψOψ; update_cache = false); alg = "loopcorrections", max_configuration_size = 16)

    denom = scalar(ψIψ; alg = "loopcorrections", max_configuration_size = 16)

    @show pupdown + pupup
    @show denom
    szsz = 2*(pupup - pupdown) / denom

    @show pupup
    @show pupdown
    @show szsz

    @show expect(ψ, obs; alg = "bp")
    # nx, ny = 4,4
    # g =named_hexagonal_lattice_graph(nx, ny; periodic = true)

    # nqubits = length(vertices(g))
    # s = ITN.siteinds("S=1/2", g)
    # ψ = ITensorNetwork(v -> "X+", s)

    # maxdim, cutoff = 4, 1e-12
    # apply_kwargs = (; maxdim, cutoff, normalize = true)
    # #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these
    # set_global_bp_update_kwargs!(;
    #     maxiter = 30,
    #     tol = 1e-10,
    #     message_update_kwargs = (;
    #         message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))
    #     ),
    # )

    # Jx, Jy, Jz = 1.0, 1.0, 1.0
    # no_eras = 6
    # ec = edge_color(g, 3)
    # xx_observables, yy_observables, zz_observables = hexagonal_kitaev_observables(Jx, Jy, Jz, ec)
    # layer_generating_function = δβ -> hexagonal_kitaev_layer(Jx, Jy, Jz, δβ, ec)
    # obs = [xx_observables; yy_observables; zz_observables]
    # energy_calculation_function = ψψ -> sum(real.(expect(ψψ, obs)))

    # ψ, ψψ = imaginary_time_evolution(ψ, layer_generating_function, energy_calculation_function, no_eras; apply_kwargs);

    # zzs = expect(ψψ, zz_observables)
    # @show zzs
    
end

main()