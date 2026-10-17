# The active tensor backend. Swapping backends means changing this file (and the backend's
# compatibility shim, included at the end of the module) and nothing else: see the header of
# src/tensor_interface.jl for what a backend implements, and rule 6 for the two type names bound here.
include("ITensorsBackend.jl")
using .ITensorsBackend: Tensor, Index
