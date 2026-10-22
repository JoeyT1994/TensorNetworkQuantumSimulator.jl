# The active tensor backend; swapping backends means changing this file and its compatibility
# shim (src/Backends/Compat.jl). See src/tensor_interface.jl for what a backend implements.
include("ITensorsBackend.jl")
using .ITensorsBackend: Tensor, Index
