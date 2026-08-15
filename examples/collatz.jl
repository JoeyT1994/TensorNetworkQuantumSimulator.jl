using TensorNetworkQuantumSimulator

using Statistics

function main()
    nx = 4
    ny = 4

    # the graph is your main friend. This will be the geometry of the TN you wull work with
    g = named_grid((4,1))
    s = siteinds("S=1/2", g)
    ψ = tensornetworkstate(v -> isodd(sum(v)) ? "Z+" : "Z-", g, s)
    gates = [("Rzz", [(1,1), (3,1)], 0.5)]

    ψ, _ = apply_gates(gates, ψ; apply_kwargs = (;maxdim = 4))
end

main()
