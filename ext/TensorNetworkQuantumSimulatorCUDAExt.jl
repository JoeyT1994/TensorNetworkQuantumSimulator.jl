module TensorNetworkQuantumSimulatorCUDAExt

using CUDA: CUDA, CuArray
using LinearAlgebra: Transpose
import TensorNetworkQuantumSimulator as TNQS

# One batched launch in place of the CPU fallback's loop over trailing slices. `B` is the same for
# every slice, and `gemm_strided_batched!` derives its strides from the arrays, so passing `B` with
# a singleton third dimension is what gives it stride 0.
function TNQS.mul_strided_batched!(C::CuArray, A::CuArray, B::Transpose)
    return TNQS.mul_strided_batched!('T', C, A, B.parent)
end
function TNQS.mul_strided_batched!(C::CuArray, A::CuArray, B::CuArray)
    return TNQS.mul_strided_batched!('N', C, A, B)
end
function TNQS.mul_strided_batched!(
        opB::Char,
        C::CuArray{T, 3}, # lead, chi, trail
        A::CuArray{T, 3}, # lead, chi, trail
        B::CuArray{T, 2}, # chi, chi
    ) where {T}

    chi = size(C, 2)

    CUDA.CUBLAS.gemm_strided_batched!(
        'N', opB,
        one(T),
        A,
        reshape(B, chi, chi, 1),
        zero(T),
        C,
    )

    return C
end

end
