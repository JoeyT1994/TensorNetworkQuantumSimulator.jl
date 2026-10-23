# The active tensor backend; swapping backends means changing this file and the compatibility
# shim included at the end of src/TensorNetworkQuantumSimulator.jl. See src/tensor_interface.jl
# for what a backend implements.
include("ITensorsBackend.jl")
using .ITensorsBackend: Tensor, Index
