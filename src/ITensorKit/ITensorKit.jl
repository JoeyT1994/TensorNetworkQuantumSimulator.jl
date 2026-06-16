module ITensorKit


using Random
using LinearAlgebra
using TensorKit
using TensorOperations
using VectorInterface
using MatrixAlgebraKit
using Adapt
using EinExprs: EinExprs, EinExpr, einexpr, SizedEinExpr
using ITensors: ITensors, Algorithm, @Algorithm_str

include("index.jl")
include("itensor.jl")
include("constructors.jl")
include("scalar.jl")
include("combiner.jl")
include("directsum.jl")
include("shims.jl")
include("contraction_sequences.jl")
include("contract.jl")
include("vectorinterface.jl")
include("factorizations.jl")
include("adapt.jl")
include("opcatalogue.jl")

# Re-export the TensorKit accessors that downstream code uses directly.
export scalartype, storagetype, spacetype, dim, space


end # module ITensorKit
