module TensorNetworkQuantumSimulatorCUDAExt

using TensorNetworkQuantumSimulator
using CUDA
using cuTENSOR
using LinearAlgebra: Diagonal, diag, mul!, rmul!
using MatrixAlgebraKit: CUSOLVER, diagview, eigh_full, gaugefix!, geqrf!,
    project_hermitian!, qr_householder!, ungqr!, uppertriangular!
using StridedViews: StridedViews
using TensorOperations: TensorOperations

const Tensors = TensorNetworkQuantumSimulator.Tensors

function _release_solver_workspace!()
    #CUDA.jl caches cuSOLVER scratch on the dense handle. The result arrays own their
    #storage, so retaining this scratch only raises the next hot path's resident baseline.
    resize!(CUDA.cuSOLVER.dense_handle().workspace_gpu, 0)
    return nothing
end

function _qr_inplace!(A)
    k = min(size(A)...)
    A, tau = geqrf!(CUSOLVER(), A)
    R = similar(A, k, size(A, 2))
    copyto!(R, view(A, axes(R)...))
    uppertriangular!(R)
    Rd = copy(diagview(R))
    Q = ungqr!(CUSOLVER(), A, tau)
    gaugefix!(qr_householder!, Q, R, Rd)
    _release_solver_workspace!()
    return Q, R
end

#Low-workspace right polar split for the tall matrices in simple update. The permuted
#workspace `A` and consumed tensor storage `Q` provide the two F-sized arrays; all other
#storage is quadratic in the smaller matrix dimension.
function _polar_split!(A, Q)
    n = size(A, 2)
    G = similar(A, n, n)
    mul!(G, A', A)
    project_hermitian!(G)
    D, V = eigh_full(G)
    _release_solver_workspace!()
    vals = diag(D)
    scale = maximum(abs, vals; init = zero(real(eltype(A))))
    cutoff = 10 * eps(real(eltype(A))) * scale
    sqrtvals = map(x -> x > cutoff ? sqrt(x) : zero(x), vals)
    invsqrtvals = map(x -> x > cutoff ? inv(sqrt(x)) : zero(x), vals)

    mul!(Q, A, V)
    rmul!(Q, Diagonal(invsqrtvals))
    mul!(A, Q, V')
    copyto!(Q, A)

    Vscaled = similar(V)
    copyto!(Vscaled, V)
    rmul!(Vscaled, Diagonal(sqrtvals))
    mul!(G, Vscaled, V')
    project_hermitian!(G)
    return Q, G
end

function Tensors._gpu_left_orthogonalize_consuming(
        t, linds, rinds, Ap, buf, cp,
        ::TensorOperations.cuTENSORBackend, dl, dr,
    )
    Q, R = if dl >= dr
        _polar_split!(reshape(Ap, dl, dr), reshape(t.data, dl, dr))
    else
        copyto!(t.data, 1, Ap, 1, length(Ap))
        _qr_inplace!(reshape(t.data, dl, dr))
    end
    TensorOperations.tensorfree!(Ap, buf)
    TensorOperations.allocator_reset!(buf, cp)
    return Q, R
end

#Request cuTENSOR's minimum-workspace plan only for the 3F-bounded fused BP path. Generic
#TensorOperations contractions retain the backend's default speed-oriented planner.
function Tensors._fused_tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β,
        ::TensorOperations.cuTENSORBackend, allocator,
    )
    Av, Bv, Cv = StridedViews.StridedView(A), StridedViews.StridedView(B),
        StridedViews.StridedView(C)
    Ainds, Binds, Cinds = collect.(TensorOperations.contract_labels(pA, pB, pAB))
    op(v, flag) = eltype(v) <: Real || !xor(flag, v.op === conj) ?
        cuTENSOR.OP_IDENTITY : cuTENSOR.OP_CONJ
    return cuTENSOR.contract!(
        α, Av, Ainds, op(Av, conjA),
        Bv, Binds, op(Bv, conjB),
        β, Cv, Cinds, cuTENSOR.OP_IDENTITY, cuTENSOR.OP_IDENTITY;
        workspace = cuTENSOR.WORKSPACE_MIN,
    )
end

Tensors._release_storage!(A::CUDA.CuArray) = CUDA.unsafe_free!(A)

end
