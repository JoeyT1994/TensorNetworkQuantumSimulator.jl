function default_dbetas(no_eras::Int64)
    dbetas = [0.25, 0.1, 0.05, 0.01, 0.005]
    no_eras <= length(dbetas) && return dbetas[1:no_eras]
    return vcat(dbetas, [last(dbetas) * (2.0^-i) for i in 1:(no_eras - length(dbetas))])
end

function imaginary_time_evolution(ψ0::ITensorNetwork, layer_generator::Function, energy_calculator::Function, no_eras::Int64, dβs = default_dbetas(no_eras); apply_kwargs, measure_freq::Int64 = 5, max_steps_per_era::Int64=200, tol = 1e-8)

    ψ = copy(ψ0)
    ψψ = build_bp_cache(ψ)

    energy = energy_calculator(ψψ)

    for era in 1:no_eras
        dβ = dβs[era]
        println("Entering era $era. Time-step is $(dβ). Energy is $energy.")
        layer = layer_generator(dβ)
        energies = [energy]
        for i in 1:max_steps_per_era
            ψt, ψψt, errs= apply(layer, ψ, ψψ; update_every = 1, verbose = false, apply_kwargs)

            if i % measure_freq == 0
                e = energy_calculator(ψψt)
                if tol != nothing && e - last(energies) >= -tol
                    break
                end
                push!(energies, e)
            end
            ψ, ψψ = copy(ψt), copy(ψψt)
        end

        energy = last(energies)
    end

    println("Imaginary time evolution finished. Final energy was $(energy)")

    return ψ, ψψ
end