function imaginary_time_evo(
    s::IndsNetwork,
    ψ::ITensorNetwork,
    model::Function,
    dbetas::Vector{<:Tuple};
    model_params,
    bp_update_kwargs=(; maxiter=10, tol=1e-10),
    apply_kwargs=(; cutoff=1e-12, maxdim=10),
  )
    ψ = copy(ψ)
    g = underlying_graph(ψ)
  
    ℋ = filter_zero_terms(model(g; model_params...))
    ψIψ = BeliefPropagationCache(QuadraticFormNetwork(ψ))
    ψIψ = update(ψIψ; bp_update_kwargs...)
    energies = Float64[]
    e_init = sum([expect(ψ, op_to_obs(op); alg="bp", (cache!)=Ref(ψIψ)) for op in ℋ])
    push!(energies, real(e_init))
    println("Starting Imaginary Time Evolution, Initial Energy is $e_init")
    β = 0
    for (i, period) in enumerate(dbetas)
      nbetas, dβ = first(period), last(period)
      println("Entering evolution period $i , β = $β, dβ = $dβ")
      U = exp(-dβ * ℋ; alg=Trotter{2}())
      gates = Vector{ITensor}(U, s)
      regauge_freq = length(gates)
      gates = vcat(gates, reverse(gates))
      for i in 1:nbetas
        for (j, gate) in enumerate(gates)
          ψ, ψIψ = apply(gate, ψ, ψIψ; normalize=true, apply_kwargs...)
          if regauge_freq != nothing && (j % regauge_freq == 1)
            ψIψ = update(ψIψ; bp_update_kwargs...)
          end
        end
        β += dβ
        ψIψ = update(ψIψ; bp_update_kwargs...)
      end
  
      e = sum([expect(ψ, op_to_obs(op); alg="bp", (cache!)=Ref(ψIψ)) for op in ℋ])
      push!(energies, real(e))
      println("Energy following evolution period is $e")
    end
  
    return ψ, energies
  end
  
  function op_to_obs(op)
    scalar = first(op.args)
    iszero(scalar) && return 0.0
    n_ops = length(op)
    vs = site.(op[1:n_ops])
    op_strings = which_op.(op[1:n_ops])
    return (reduce(*, op_strings), vs, scalar)
  end
  
  
  function filter_zero_terms(H::OpSum)
    new_H = OpSum()
    for h in H
      if !iszero(first(h.args))
        new_H += h
      end
    end
    return new_H
  end
  