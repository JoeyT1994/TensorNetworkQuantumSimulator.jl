module TensorNetworkQuantumSimulatorCUDAExt

using CUDA: CUDA, CuMatrix
using CUDA.CUSOLVER: CUSOLVER
using LinearAlgebra: triu!
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator

# cuSOLVER names the Q-generation routine differently for real and complex.
generate_q!(A, tau) = eltype(A) <: Complex ? CUSOLVER.ungqr!(A, tau) : CUSOLVER.orgqr!(A, tau)

# Device method for the seam in src/MessagePassing/beliefpropagation_mpi.jl.
#
# This exists because `LinearAlgebra.qr!` + `lmul!(F.Q, ...)` -- and `CuMatrix(F.Q)`, which
# CUDA.jl implements via `lmul!` -- go through cuSOLVER `ormqr`, which rejects matrices with
# more than typemax(Int32) elements (CUSOLVER_STATUS_INVALID_VALUE). `orgqr`/`ungqr` accept the
# same dimensions. Verified on an RTX PRO 6000: both paths fine at χ = 512 (5.4e8 elements),
# `ormqr` fails and `orgqr` succeeds at χ = 1024 (4.3e9).
#
# Q overwrites `A`; R is copied out first, since `orgqr!` destroys the reflectors it sits among.
function TensorNetworkQuantumSimulator.thin_qr_matrix!(A::CuMatrix)
    n = size(A, 2)
    A, tau = CUSOLVER.geqrf!(A)
    R = triu!(A[1:n, :])
    generate_q!(A, tau)
    return A, R
end

end
