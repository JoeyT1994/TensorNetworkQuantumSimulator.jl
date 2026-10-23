# Forwards ITensors functions called on package objects (e.g. `ITensors.contract(ψ)`) to the
# package, for scripts written against earlier releases. Delete together with the ITensors backend.
module ITensorsCompat

using ITensors: ITensors
using ITensors.NDTensors: NDTensors

import ..TensorNetworkQuantumSimulator as TNQS
using ..TensorNetworkQuantumSimulator: AbstractTensorNetwork, AbstractBeliefPropagationCache,
    AbstractForm, TensorNetworkState

const PackageObject = Union{AbstractTensorNetwork, AbstractBeliefPropagationCache, AbstractForm}

ITensors.datatype(x::PackageObject) = TNQS.datatype(x)
NDTensors.scalartype(x::PackageObject) = TNQS.scalartype(x)
ITensors.contract(tn::AbstractTensorNetwork; kwargs...) = TNQS.contract(tn; kwargs...)
ITensors.inner(ψ::TensorNetworkState, ϕ::TensorNetworkState; kwargs...) = TNQS.inner(ψ, ϕ; kwargs...)

end
