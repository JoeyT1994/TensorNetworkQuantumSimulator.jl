# Does `CUSOLVER.Xgeqrf!` + `SOLVER.ormqr!` replace the hand-rolled TSQR in
# `src/Apply/blocked_gate.jl`?
#
# `Xgeqrf` is cuSOLVER's 64-bit QR: `int64_t` m/n/lda and `size_t` workspaces, so nothing in its
# sizing path can overflow, and it factorises in place -- no copy of M. TSQR by contrast retains
# every block's Q (`TallSkinnyQR.blocks`), and those blocks partition M's rows, so they add up to a
# full duplicate of M: ~2F where this would be ~1F.
#
# There is no 64-bit apply-Q anywhere in cuSOLVER (checked against the CUDA 12.9-13.3 docs: the
# cusolverDnX API has geqrf and larft but no Xunmqr/Xormqr/Xungqr), so applying Q still goes through
# the legacy 32-bit `ormqr`. That is the open question this probe answers. `ormqr`'s workspace is
# O(n*nb), not O(m*n) -- the documented overflow reports are all against `orgqr`, which *forms* an
# m x n Q. If `ormqr` is fine at these shapes then TSQR is working around a wall that is not there.
#
# Shapes are the production ones: a degree-3 superket vertex is S*chi^3 = 4*chi^3 elements, reshaped
# to (4*chi^2) x chi for the QR. chi=812 is the last size under 2^31 elements.
#
# Run:  julia --project -e 'include("examples/qr_64bit_probe.jl")'

using CUDA
using LinearAlgebra
using Printf

# Recent CUDA.jl split cuSOLVER into its own package (the traceback shows
# `~/.julia/packages/cuSOLVER/...`); older versions nest it as `CUDA.CUSOLVER`. Resolved rather than
# assumed, because the two layouts are not interchangeable.
const SOLVER = if isdefined(CUDA, :CUSOLVER)
    getfield(CUDA, :CUSOLVER)
else
    try
        @eval using cuSOLVER
        @eval cuSOLVER
    catch err
        error(
            "Could not find cuSOLVER: it is neither `CUDA.CUSOLVER` nor a loadable `cuSOLVER` " *
                "package in this environment. Add it with `] add cuSOLVER`. ($err)"
        )
    end
end
println("cuSOLVER module: ", SOLVER)

const T = ComplexF32

function meminfo()
    return try
        free, total = CUDA.memory_info()
        @sprintf("%.1f/%.1f GiB free", free / 2^30, total / 2^30)
    catch
        "memory_info unavailable"
    end
end

# `Xgeqrf!` overwrites A: R in the upper triangle, Householder vectors below. Reconstructing M from
# (A, tau) is the only honest check that both halves agree -- a QR that runs but returns nonsense
# would otherwise look like a pass.
function check_roundtrip(chi)
    m, n = 4 * chi^2, chi
    M0 = CUDA.randn(T, m, n)
    M = copy(M0)
    A, tau = SOLVER.Xgeqrf!(M)

    # C = [R; 0], then Q*C should be M0.
    C = CUDA.zeros(T, m, n)
    # Materialise the n x n block before `triu!`: CUDA.jl's `triu`/`triu!` are defined for a dense
    # `CuMatrix`, and a row-range view is a `SubArray` that will not dispatch to them.
    Rblk = CuArray(view(A, 1:n, :))
    triu!(Rblk)
    copyto!(view(C, 1:n, :), Rblk)
    CUDA.unsafe_free!(Rblk)
    SOLVER.ormqr!('L', 'N', A, tau, C)

    err = norm(C - M0) / norm(M0)
    CUDA.unsafe_free!(M0); CUDA.unsafe_free!(A); CUDA.unsafe_free!(C); CUDA.unsafe_free!(tau)
    return err
end

# At production chi a round-trip needs three m x n arrays at once (32 GiB each at chi=1024), so the
# large sizes only check that the two calls complete. `cwidth` is how many columns Q is applied to;
# `ormqr`'s workspace scales with it, so it is the knob that would trip an int32 lwork.
function check_runs(chi; cwidth = chi)
    m, n = 4 * chi^2, chi
    bytes = (m * n + m * cwidth) * sizeof(T)
    @printf("  chi=%-5d m=%-10d n=%-5d  m*n=%.3e  needs ~%.1f GiB  [%s]\n",
        chi, m, n, m * n, bytes / 2^30, meminfo())
    m * n > typemax(Int32) && println("    (m*n is past 2^31 -- this is the case TSQR exists for)")

    M = CUDA.randn(T, m, n)
    A, tau = SOLVER.Xgeqrf!(M)
    CUDA.synchronize()
    println("    Xgeqrf! ok")

    C = CUDA.randn(T, m, cwidth)
    SOLVER.ormqr!('L', 'N', A, tau, C)
    CUDA.synchronize()
    println("    ormqr!  ok")

    CUDA.unsafe_free!(A); CUDA.unsafe_free!(C); CUDA.unsafe_free!(tau)
    return nothing
end

println("device: ", CUDA.name(CUDA.device()), "  ", meminfo())
println("CUDA runtime: ", CUDA.runtime_version())

println("\n=== correctness (small, full round-trip) ===")
for chi in (32, 64, 128)
    err = try
        @sprintf("%.2e", check_roundtrip(chi))
    catch e
        "FAILED: $(sprint(showerror, e))"
    end
    println("  chi=", rpad(chi, 5), " ||QR - M||/||M|| = ", err)
end

println("\n=== scaling (does it survive past 2^31 elements?) ===")
for chi in (512, 800, 900, 1024)
    try
        check_runs(chi)
    catch e
        println("    FAILED at chi=$chi: ", sprint(showerror, e))
        break     # a sticky CUDA error poisons the context; later sizes would be meaningless
    end
    GC.gc(); CUDA.reclaim()
end
