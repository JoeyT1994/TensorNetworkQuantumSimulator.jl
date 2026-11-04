# The active tensor backend; swapping backends means changing this file. See src/tensor_interface.jl
# for what a backend implements.
include("ITensorBaseBackend.jl")
using .ITensorBaseBackend: ITensorBaseBackend, Index, AbstractTensor, AnyTensor

# The package's tensor type, for annotations and containers.
const Tensor = AnyTensor
