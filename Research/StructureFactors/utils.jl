function honeycomb_kitaev_layer(Jx::Float64, Jy::Float64, Jz::Float64, δβ::Float64, ec)
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

function honeycomb_kitaev_observables(Jx::Float64, Jy::Float64, Jz::Float64, ec)
    xx_observables = [("XX", pair, Jx) for pair in ec[1]]
    yy_observables = [("YY", pair, Jy) for pair in ec[2]]
    zz_observables = [("ZZ", pair, Jz) for pair in ec[3]]
    return xx_observables, yy_observables, zz_observables
end

function ising_layer(J::Float64, h::Float64, δβ::Float64, ec)
    es = reduce(vcat, ec)
    vs = unique(vcat(first.(es), last.(es)))
    layer = []

    append!(layer, ("Rx", [v], -h*δβ * im) for v in vs)
    append!(layer, ("Rzz", pair, -2*J*δβ * im) for pair in es)
    append!(layer, ("Rx", [v], -h*δβ * im) for v in vs)

    return layer
end

function ising_observables(J::Float64, h::Float64, ec)
    es = reduce(vcat, ec)
    vs = unique(vcat(first.(es), last.(es)))
    zz_observables = [("ZZ", pair, J) for pair in reduce(vcat, ec)]
    x_observables = [("X", [v], h) for v in vs]
    return x_observables, zz_observables
end

function honeycomb_kitaev_heisenberg_layer(J::Float64, K::Float64, δβ::Float64, ec)
    layer = []
    append!(layer, ("RxxRyyRzz", pair, -(K + J)*δβ * im, -(J)*δβ * im, -(J)*δβ * im) for pair in ec[1])
    append!(layer, ("RxxRyyRzz", pair, -(J)*δβ * im, -(K + J)*δβ * im, -(J)*δβ * im) for pair in ec[2])
    append!(layer, ("RxxRyyRzz", pair, -2*(J)*δβ * im, -2*(J)*δβ * im, -2*(K + J)*δβ * im) for pair in ec[3])
    append!(layer, ("RxxRyyRzz", pair, -(J)*δβ * im, -(K + J)*δβ * im, -(J)*δβ * im) for pair in ec[2])
    append!(layer, ("RxxRyyRzz", pair, -(K + J)*δβ * im, -(J)*δβ * im, -(J)*δβ * im) for pair in ec[1])
    return layer
end

function honeycomb_kitaev_heisenberg_realtime_layer(J::Float64, K::Float64, δt::Float64, ec)
    layer = []
    append!(layer, ("RxxRyyRzz", pair, (K + J)*δt, (J)*δt, (J)*δt) for pair in ec[1])
    append!(layer, ("RxxRyyRzz", pair, (J)*δt, (K + J)*δt, (J)*δt) for pair in ec[2])
    append!(layer, ("RxxRyyRzz", pair, 2*(J)*δt, 2*(J)*δt, 2*(K + J)*δt) for pair in ec[3])
    append!(layer, ("RxxRyyRzz", pair, (J)*δt, (K + J)*δt, (J)*δt) for pair in ec[2])
    append!(layer, ("RxxRyyRzz", pair, (K + J)*δt, (J)*δt, (J)*δt) for pair in ec[1])
    return layer
end


function honeycomb_kitaev_heisenberg_observables(J::Float64, K::Float64, ec)
    xx_observables = vcat([("XX", pair, (J + K)) for pair in ec[1]], [("XX", pair, (J)) for pair in vcat(ec[2], ec[3])])
    yy_observables = vcat([("YY", pair, (J + K)) for pair in ec[2]], [("YY", pair, (J)) for pair in vcat(ec[1], ec[3])])
    zz_observables = vcat([("ZZ", pair, (J + K)) for pair in ec[3]], [("ZZ", pair, (J)) for pair in vcat(ec[1], ec[2])])
    return xx_observables, yy_observables, zz_observables
end