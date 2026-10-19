#=
ITensorsCompat: ITensors-facing compatibility for code written against the ITensors-backed
releases, kept here so the rest of the package stays backend-agnostic. Delete it together with
the ITensors backend.

Before the seam existed the package extended ITensors' own functions, so user scripts (and some
tests and examples) call them on package objects — `ITensors.contract(ψ; alg = "exact")`,
`datatype(ψ)` imported from ITensors — or build the package's in-house gates through
`ITensors.op("Rxxyy", s1, s2; θ)`. These methods forward such calls to the package.
=#
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
