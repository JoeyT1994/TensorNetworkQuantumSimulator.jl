function default_schedule(no_eras::Int64)
    no_steps = 1000
    schedule = [(0.25, 50), (0.1, 250), (0.05, no_steps),  (0.01, no_steps),  (0.005, no_steps)]
    no_eras <= length(schedule) && return schedule[1:no_eras]
    return vcat(schedule, [(first(last(schedule)) * (2.0^-i), no_steps) for i in 1:(no_eras - length(schedule))])
end

function imaginary_time_evolution(ψ0::ITensorNetwork, layer_generator::Function, energy_calculator::Function, no_eras::Int64, schedule = default_schedule(no_eras); apply_kwargs)

    ψ = copy(ψ0)
    ψψ = build_bp_cache(ψ)

    for (era, (dβ, no_steps)) in enumerate(schedule)
        energy = energy_calculator(ψψ)
        println("Entering era $era. Time-step is $(dβ). No of steps is $(no_steps). Energy is $energy.")
        layer = layer_generator(dβ)
        for i in 1:no_steps
            ψ, ψψ, errs= apply(layer, ψ, ψψ; apply_kwargs)
        end
        flush(stdout)
    end

    energy = energy_calculator(ψψ)

    println("Imaginary time evolution finished. Final energy was $(energy)")

    return ψ, ψψ, energy
end