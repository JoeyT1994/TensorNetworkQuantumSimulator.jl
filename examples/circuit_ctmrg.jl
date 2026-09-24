# A 1+1D quantum circuit contracted as a space–time rectangle with finite CTMRG: space and time on
# equal footing, the corners compressing the light cone from all four sides (boundary MPS would
# sweep in time and pay χ for the spatial entanglement only).
#
#   julia -t 8 --project examples/circuit_ctmrg.jl
#
# The circuit is a brickwork of random two-qubit unitaries on N qubits, DEPTH layers, from |0…0⟩.
# Two quantities, both scalar networks on the (qubit, layer) grid:
#   1. the return amplitude ⟨0…0|U|0…0⟩ — one layer, D = 2 vertical bonds, gate halves on the
#      horizontal bonds (rank ≤ 4);
#   2. ⟨Z_k⟩ on the final state — the folded double layer (ket ⊗ conj ket), D = 4 bonds.
# Both are checked against a statevector simulation. The network is non-Hermitian (no positivity),
# which is what the `:cycle` (eig-CTMRG) projector is for; `:cut` (svd-CTMRG) is run alongside.
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
const TI = TNQS.TensorInterface
using TensorNetworkQuantumSimulator.TensorInterface: new_index, from_array
using Dictionaries: Dictionary, set!
using LinearAlgebra, Random, Printf

# ── parameters ───────────────────────────────────────────────────────────────────────────
# Every constant can be overridden by an environment variable CIRC_<NAME> (lists comma-separated),
# so one script serves a Slurm/disBatch sweep (examples/slurm/circuit_sweep.sbatch).
_env(name, default) = get(ENV, "CIRC_" * name, default)
_envlist(name, default, f) = Tuple(f.(strip.(split(_env(name, default), ","; keepempty = false))))
const N          = parse(Int, _env("N", "10"))            # qubits (statevector reference: keep ≤ 22)
const DEPTH      = parse(Int, _env("DEPTH", "8"))         # brickwork layers (for the ⟨Z⟩ test)
const CHIS       = _envlist("CHIS", "4,8,16,32", x -> parse(Int, x))       # CTMRG interface dimensions
const PROJECTORS = _envlist("PROJECTORS", "cut,cycle", Symbol)             # svd-CTMRG and eig-CTMRG
const ZSITE      = parse(Int, _env("ZSITE", "5"))         # ⟨Z_k⟩ is measured on this qubit
const MAXITER    = parse(Int, _env("MAXITER", "60"))
const SEED       = parse(Int, _env("SEED", "1"))
const GATESET    = Symbol(_env("GATESET", "haar"))        # :haar (random two-qubit unitaries) or :sycamore
                                                          #   (fSim(π/2, π/6) couplers preceded by random √X, √Y, √W)
const DEPTHS     = _envlist("DEPTHS", "", x -> parse(Int, x))   # non-empty: amplitude-vs-depth sweep at N
                                                          #   qubits instead of the two tests, e.g. "2,4,8,12"
Random.seed!(SEED)
function haar(n)                                             # Haar-random unitary (QR with phase fix)
    F = qr(randn(ComplexF64, n, n))
    return Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
end
# Sycamore's coupler fSim(θ, φ) on (|00⟩,|01⟩,|10⟩,|11⟩) and its single-qubit gates √X, √Y, √W
fsim(θ, φ) = ComplexF64[1 0 0 0; 0 cos(θ) -im*sin(θ) 0; 0 -im*sin(θ) cos(θ) 0; 0 0 0 exp(-im*φ)]
const SQX = ComplexF64[1 -im; -im 1] / sqrt(2)
const SQY = ComplexF64[1 -1; 1 1] / sqrt(2)
const SQW = ComplexF64[1 -sqrt(im); sqrt(-im) 1] / sqrt(2)
# One brickwork layer's two-qubit gates: (layer, left qubit) => 4×4 unitary. For :sycamore every
# qubit gets a random single-qubit gate (never the same as its previous one) BEFORE the coupler
# layer; it is absorbed into the coupler of the pair, or stands alone on an idle qubit.
function build_gates(depth)
    gates = Dict{Tuple{Int, Int}, Matrix{ComplexF64}}()
    singles = Dict{Tuple{Int, Int}, Matrix{ComplexF64}}()     # (layer, qubit) => 2×2 on idle qubits
    last = fill(0, N)
    for y in 1:depth
        one = Dict{Int, Matrix{ComplexF64}}()
        if GATESET === :sycamore
            for x in 1:N
                c = rand(setdiff(1:3, last[x])); last[x] = c
                one[x] = (SQX, SQY, SQW)[c]
            end
        end
        covered = falses(N)
        for x in (isodd(y) ? 1 : 2):2:(N - 1)
            G = GATESET === :sycamore ? fsim(π / 2, π / 6) * kron(one[x + 1], one[x]) : haar(4)
            gates[(y, x)] = G; covered[x] = covered[x + 1] = true
        end
        GATESET === :sycamore && for x in 1:N
            covered[x] || (singles[(y, x)] = one[x])
        end
    end
    return gates, singles
end
gates, singles = build_gates(DEPTH)
println("brickwork circuit ($GATESET): N = $N, depth = $DEPTH, $(length(gates)) two-qubit gates, seed $SEED")

# ── statevector reference ────────────────────────────────────────────────────────────────
function apply_gate!(ψ::Vector{ComplexF64}, G::Matrix{ComplexF64}, x::Int, N::Int)
    # qubit x is the x-th factor (x = 1 fastest); apply G to qubits (x, x+1)
    ψr = reshape(ψ, 2^(x - 1), 4, 2^(N - x - 1))
    out = similar(ψr)
    for k in axes(ψr, 3), i in axes(ψr, 1)
        out[i, :, k] = G * ψr[i, :, k]
    end
    return vec(out)
end
function apply_single!(ψ::Vector{ComplexF64}, g::Matrix{ComplexF64}, x::Int, N::Int)
    ψr = reshape(ψ, 2^(x - 1), 2, 2^(N - x))
    out = similar(ψr)
    for k in axes(ψr, 3), i in axes(ψr, 1)
        out[i, :, k] = g * ψr[i, :, k]
    end
    return vec(out)
end
function apply_layer(ψ, gates, singles, y, N)
    for x in (isodd(y) ? 1 : 2):2:(N - 1)
        ψ = apply_gate!(ψ, gates[(y, x)], x, N)
    end
    for x in 1:N
        haskey(singles, (y, x)) && (ψ = apply_single!(ψ, singles[(y, x)], x, N))
    end
    return ψ
end
function statevector(gates, singles, depth)
    ψ = zeros(ComplexF64, 2^N); ψ[1] = 1
    for y in 1:depth
        ψ = apply_layer(ψ, gates, singles, y, N)
    end
    return ψ
end
ψsv = statevector(gates, singles, DEPTH)
amp_exact = ψsv[1]
zk(ψ, k) = sum(abs2(ψ[i]) * (iszero((i - 1) >> (k - 1) & 1) ? 1 : -1) for i in eachindex(ψ))
z_exact = zk(ψsv, ZSITE)
@printf("statevector: |⟨0|U|0⟩| = %.10e   ln|A| = %.10f   ⟨Z_%d⟩ = %.10f\n", abs(amp_exact), log(abs(amp_exact)), ZSITE, z_exact)

# ── the space–time rectangle ─────────────────────────────────────────────────────────────
# Gate (y, x) = Σ_k A_k(x) ⊗ B_k(x+1) by SVD (operator-Schmidt), rank r ≤ 4; the two halves carry
# a horizontal bond of dimension r. Site (x, y) tensor: legs (in, out, [left bond], [right bond]).
# A qubit idle in a layer gets the identity on (in, out). Layers are stacked by sharing the vertical
# index: out of (x, y) = in of (x, y+1). The bottom in-legs are contracted with |0⟩, the top
# out-legs with the boundary vector `top[x]` (⟨0| for the amplitude; for the folded network see below).
function split_gate(G)
    M = reshape(permutedims(reshape(G, 2, 2, 2, 2), (1, 3, 2, 4)), 4, 4)   # (out1 in1) × (out2 in2)
    F = svd(M)
    r = count(s -> s > 1e-12, F.S)
    A = reshape(F.U[:, 1:r] * Diagonal(sqrt.(F.S[1:r])), 2, 2, r)        # (out1, in1, k)
    B = reshape(Diagonal(sqrt.(F.S[1:r])) * F.Vt[1:r, :], r, 2, 2)        # (k, out2, in2)
    return A, permutedims(B, (2, 3, 1))                                   # both (out, in, k)
end
# `fold = false`: the single layer with local dimension 2. `fold = true`: T ⊗ conj(T) with every leg
# fused (dimension d²), for ⟨ψ|O|ψ⟩; `top[x]` is then the vectorised operator on qubit x.
function circuit_network(gates, singles, N, DEPTH; fold::Bool = false, top = nothing)
    d = fold ? 4 : 2
    vert = Dict{Tuple{Int, Int}, Any}()                    # (x, y) => index between layer y and y+1
    for x in 1:N, y in 0:DEPTH
        vert[(x, y)] = new_index(d; tags = "v$(x)_$(y)")
    end
    # T ⊗ conj(T) with every leg fused to (ket, bra), ket fastest — the same order as the boundary
    # vectors below (|0⟩⟨0| = [1,0,0,0], tr = [1,0,0,1], Z = [1,0,0,-1])
    function fuse(A)
        n = ndims(A)
        P = reshape([a * conj(b) for a in vec(A), b in vec(A)], size(A)..., size(A)...)
        return reshape(permutedims(P, vcat([[i, n + i] for i in 1:n]...)), ntuple(i -> size(A, i)^2, n)...)
    end
    lay(A) = fold ? fuse(A) : A
    tensors = Dictionary{Tuple{Int, Int}, Any}()
    for y in 1:DEPTH
        halves = Dict{Int, Any}()                          # x => (array (out, in, k), bond index, side)
        for x in (isodd(y) ? 1 : 2):2:(N - 1)
            A, B = split_gate(gates[(y, x)])
            r = size(A, 3)
            k = new_index(fold ? r^2 : r; tags = "h$(x)_$(y)")
            halves[x] = (lay(A), k); halves[x + 1] = (lay(B), k)
        end
        for x in 1:N
            iin, iout = vert[(x, y - 1)], vert[(x, y)]
            if haskey(halves, x)
                A, k = halves[x]
                t = from_array(A, iout, iin, k)
            else
                g = get(singles, (y, x), Matrix{ComplexF64}(I, 2, 2))
                t = from_array(lay(g), iout, iin)
            end
            # boundaries: |0⟩ (or |0⟩⟨0| folded) at the bottom, the boundary vector at the top
            if y == 1
                v0 = fold ? ComplexF64[1, 0, 0, 0] : ComplexF64[1, 0]
                t = t * from_array(v0, iin)
            end
            if y == DEPTH
                vt = top === nothing ? (fold ? ComplexF64[1, 0, 0, 1] : ComplexF64[1, 0]) : top[x]
                t = t * from_array(vt, iout)
            end
            set!(tensors, (x, y), t)
        end
    end
    return TensorNetwork(tensors)
end

function ctm_lnZ(tn, χ, projector)
    cache = CTMEnvironmentCache(tn, χ; projector)
    τ = @elapsed cache = update(cache; maxiter = MAXITER)
    return cvm_freenergy(cache), τ
end

# ── amplitude-vs-depth sweep ─────────────────────────────────────────────────────────────
if !isempty(DEPTHS)
    gates_all, singles_all = build_gates(maximum(DEPTHS))
    ψ = zeros(ComplexF64, 2^N); ψ[1] = 1
    println("\namplitude ⟨0…0|U|0…0⟩ vs depth, $GATESET gates, N = $N: Δ ln|A| per projector and χ")
    @printf("%6s %14s", "depth", "ln|A| exact")
    for projector in PROJECTORS, χ in CHIS; @printf(" %12s", "$(projector)/$χ"); end
    println()
    yprev = 0
    for depth in DEPTHS
        global ψ, yprev
        for y in (yprev + 1):depth
            ψ = apply_layer(ψ, gates_all, singles_all, y, N)
        end
        yprev = depth
        lnA = log(abs(ψ[1]))
        @printf("%6d %14.8f", depth, lnA)
        tn = circuit_network(gates_all, singles_all, N, depth)
        for projector in PROJECTORS, χ in CHIS
            F, _ = ctm_lnZ(tn, χ, projector)
            @printf(" %12.2e", F - lnA)
        end
        println(); flush(stdout)
    end
    exit()
end

# 1. the return amplitude (single layer): ln|A|
tn_amp = circuit_network(gates, singles, N, DEPTH)
println("\n1. return amplitude, single layer (D = 2), exact ln|A| = $(log(abs(amp_exact)))")
lnA_tn = log(abs(contract(tn_amp; alg = "exact")))
@printf("   network exact contraction: ln|A| = %.10f (Δ %.1e vs statevector)\n", lnA_tn, lnA_tn - log(abs(amp_exact)))
for projector in PROJECTORS, χ in CHIS
    F, τ = ctm_lnZ(tn_amp, χ, projector)
    @printf("   %-6s χ = %2d: ln|A| = %.10f   Δ = %+.2e   (%.1f s)\n", projector, χ, F, F - log(abs(amp_exact)), τ)
    flush(stdout)
end

# 2. ⟨Z_k⟩ on the final state: folded network, two contractions (Z at k / identity)
Zvec = ComplexF64[1, 0, 0, -1]; Ivec = ComplexF64[1, 0, 0, 1]
topZ = Dict(x => (x == ZSITE ? Zvec : Ivec) for x in 1:N)
tn_Z = circuit_network(gates, singles, N, DEPTH; fold = true, top = topZ)
tn_I = circuit_network(gates, singles, N, DEPTH; fold = true)
println("\n2. ⟨Z_$ZSITE⟩, folded double layer (D = 4), exact $(z_exact)")
zI = contract(tn_I; alg = "exact"); zZ = contract(tn_Z; alg = "exact")
@printf("   network exact contraction: norm %.6f  ⟨Z⟩ = %.10f (Δ %.1e vs statevector)\n", real(zI), real(zZ / zI), real(zZ / zI) - z_exact)
for projector in PROJECTORS, χ in CHIS
    FZ, τ1 = ctm_lnZ(tn_Z, χ, projector); FI, τ2 = ctm_lnZ(tn_I, χ, projector)
    # cvm_freenergy is ln|z| per region: the sign of ⟨Z⟩ is fixed by the exact value's sign here
    z = sign(real(zZ)) * exp(FZ - FI)
    @printf("   %-6s χ = %2d: ⟨Z⟩ = %+.10f   Δ = %+.2e   (%.1f s)\n", projector, χ, z, z - z_exact, τ1 + τ2)
    flush(stdout)
end
