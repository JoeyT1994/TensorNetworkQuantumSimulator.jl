# Forwards ITensors functions called on package objects (e.g. `ITensors.contract(ψ)`,
# `ITensors.op("Rxxyy", s1, s2; θ)`) to the package, for scripts written against earlier
# releases. Delete together with the ITensors backend.
module ITensorsCompat

using ITensors: ITensors, Index
using ITensors.NDTensors: NDTensors
using ITensors.SiteTypes: OpName, SiteType

import ..TensorNetworkQuantumSimulator as TNQS
using ..TensorNetworkQuantumSimulator: AbstractTensorNetwork, AbstractBeliefPropagationCache,
    AbstractForm, TensorNetworkState

const PackageObject = Union{AbstractTensorNetwork, AbstractBeliefPropagationCache, AbstractForm}

ITensors.datatype(x::PackageObject) = TNQS.datatype(x)
NDTensors.scalartype(x::PackageObject) = TNQS.scalartype(x)
ITensors.contract(tn::AbstractTensorNetwork; kwargs...) = TNQS.contract(tn; kwargs...)
ITensors.inner(ψ::TensorNetworkState, ϕ::TensorNetworkState; kwargs...) = TNQS.inner(ψ, ϕ; kwargs...)

# The in-house two-site gates under their ITensors names.
for name in ("Rxxyy", "Rxxyyzz", "xx_plus_yy"), st in ("S=1/2", "Qubit")
    @eval ITensors.op(::$(typeof(OpName(name))), ::$(typeof(SiteType(st))), s1::Index, s2::Index; kwargs...) =
        TNQS.op($name, s1, s2; kwargs...)
end

end
