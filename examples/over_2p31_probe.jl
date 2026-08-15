# Which GPU operations break on an array with more than 2^31 ELEMENTS?
#
#     julia --project=<env with CUDA> over_2p31_probe.jl
#
# A degree-3 tensor-network vertex at S=4 is S*chi^3 elements: 2.05e9 at chi=800 (under 2^31) and
# 4.29e9 at chi=1024 (2.0x over). Runs at chi=800 succeed and chi=1024 dies with
# CUDA_ERROR_ILLEGAL_ADDRESS, so the hypothesis is a 32-bit linear index somewhere in the stack.
#
# Footprints at chi=1024, ComplexF32:
#   fill!, sum          1 array    32 GiB
#   permutedims!, mul!  2 arrays   64 GiB
#
# An out-of-memory failure is caught and reported like any other, and reads quite differently from
# an illegal access, so nothing here needs to know the card's capacity in advance. (The
# memory-query API has moved between CUDA.jl generations, which is why this does not ask.)
#
# `CUDA.@sync` is load-bearing: CUDA reports errors asynchronously, so without it an illegal access
# surfaces at whatever the next synchronisation happens to be. That is exactly how this bug first
# presented -- as a failure inside an unrelated `sum`, several calls after the kernel at fault.
using CUDA, LinearAlgebra, Printf

CUDA.versioninfo()
@printf("\n2^31 = %d elements\n\n", 2^31)

function attempt(label, chi, f)
    n = 4 * chi^3
    msg = try
        CUDA.@sync f(chi)
        "ok"
    catch e
        "FAIL: " * first(split(sprint(showerror, e), '\n'))
    end
    @printf("  %-14s chi=%-5d %11d elements  %-6s %s\n",
            label, chi, n, n > 2^31 ? "OVER" : "under", msg)
    flush(stdout)
    GC.gc()
    CUDA.reclaim()
    return
end

function do_fill(chi)
    A = CUDA.zeros(ComplexF32, 4 * chi^3)
    fill!(A, 1.0f0)
    return A = nothing
end

function do_sum(chi)
    A = CUDA.ones(ComplexF32, 4 * chi^3)
    s = sum(A)
    A = nothing
    return s
end

# The aligned site tensor, permuted the way the blocked message kernel permutes it.
function do_perm(chi)
    A = CUDA.randn(ComplexF32, 4, chi, chi, chi)
    B = CUDA.zeros(ComplexF32, chi, 4, chi, chi)
    permutedims!(B, A, (2, 1, 3, 4))
    return A = B = nothing
end

# chi^2 x S*chi, the shape the gate's QR and the kernel's closing gemm both see.
function do_mul(chi)
    A = CUDA.randn(ComplexF32, chi * chi, 4 * chi)
    B = CUDA.randn(ComplexF32, 4 * chi, 8)
    C = CUDA.zeros(ComplexF32, chi * chi, 8)
    mul!(C, A, B)
    return A = B = C = nothing
end

for (label, f) in (("fill!", do_fill), ("sum", do_sum),
                   ("permutedims!", do_perm), ("mul!", do_mul))
    println("$label:")
    for chi in (640, 800, 1024)
        attempt(label, chi, f)
    end
    println()
end
