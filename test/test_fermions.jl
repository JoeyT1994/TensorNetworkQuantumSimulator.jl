@eval module $(gensym())
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test
using Random
using ITensors
using ITensors: ITensors, Index, ITensor, dim, inds
using Dictionaries: Dictionary, set!

const FT = TN.FermionicTensor

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# random parity-even ITensor over indices `is` w.r.t. grading `gr`
function rand_even(is::Vector{<:Index}, gr::Dictionary)
    bits = [gr[i] for i in is]
    dims = ntuple(k -> dim(is[k]), length(is))
    arr = zeros(ComplexF64, dims...)
    for I in CartesianIndices(dims)
        odd = false
        for k in 1:length(is)
            odd ⊻= bits[k][I[k]]
        end
        odd || (arr[I] = randn(ComplexF64))
    end
    return ITensor(arr, is...)
end

# Independent brute-force reorder oracle: transform leg order `from -> to` by
# adjacent bubble swaps, each adjacent swap of legs (a,b) multiplying the data by
# (−1)^{bit_a bit_b}. A genuinely different code path from the fast inversion-count
# `_reorder_sign`, so agreement validates the sign bookkeeping.
function brute_reorder(gr::Dictionary, T::ITensor, from::Vector{<:Index}, to::Vector{<:Index})
    cur = collect(from)
    arr = ITensors.array(T, cur...)
    tp = Dict(ind => p for (p, ind) in enumerate(to))
    n = length(cur)
    swapped = true
    while swapped
        swapped = false
        for p in 1:(n - 1)
            if tp[cur[p]] > tp[cur[p + 1]]
                ba, bb = gr[cur[p]], gr[cur[p + 1]]
                sgn = [(xa && xb) ? -1.0 : 1.0 for xa in ba, xb in bb]
                shape = ntuple(d -> d == p ? length(ba) : (d == p + 1 ? length(bb) : 1), n)
                arr = arr .* reshape(sgn, shape)
                perm = collect(1:n); perm[p], perm[p + 1] = p + 1, p
                arr = permutedims(arr, Tuple(perm))
                cur[p], cur[p + 1] = cur[p + 1], cur[p]
                swapped = true
            end
        end
    end
    return ITensor(arr, to...)
end

fscalar(ft::FT) = ITensors.scalar(ft.tensor)

@testset "Test fermions" begin
    Random.seed!(1234)

    @testset "constructor" begin
        g = named_grid((2, 3))
        s = siteinds("fermion", g)
        ψ = random_fermionic_tensornetworkstate(g, s; bond_dimension = 2)
        @test length(vertices(ψ)) == 6
        for v in vertices(ψ)
            ft = FT(ψ, v)
            @test Set(ft.order) == Set(inds(ψ[v]))
            @test length(ft.dirs) == length(ft.order)
        end
    end

    @testset "fermionic_transpose vs brute oracle" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mk(d, bits) = (i = Index(d, "x"); set!(gr, i, bits); i)
        for _ in 1:20
            is = [mk(2, [false, true]), mk(3, [false, true, true]), mk(2, [false, true]), mk(2, [true, false])]
            T = rand_even(is, gr)
            dirs = rand(Bool, length(is))
            ft = FT(T, copy(is), dirs, gr)
            to = shuffle(is)
            fast = fermionic_transpose(ft, to)
            brute = brute_reorder(gr, T, is, to)
            @test fast.tensor ≈ brute
            @test fast.order == to
            # dirs travel with their legs
            @test all(fast.dirs[k] == dirs[findfirst(==(to[k]), is)] for k in 1:length(is))
        end
    end

    @testset "fermionic_dag is an involution" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mk(d, bits) = (i = Index(d, "x"); set!(gr, i, bits); i)
        is = [mk(2, [false, true]), mk(2, [false, true]), mk(3, [false, true, true])]
        T = rand_even(is, gr)
        ft = FT(T, copy(is), rand(Bool, 3), gr)
        dd = fermionic_dag(fermionic_dag(ft))
        @test dd.order == ft.order
        @test dd.dirs == ft.dirs
        @test dd.tensor ≈ ft.tensor
    end

    @testset "swap rule: binary contract is operand-order independent (even tensors)" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mkbond() = (i = Index(2, "b"); set!(gr, i, [false, true]); i)
        for _ in 1:20
            bond = mkbond()
            ao = mkbond(); bo = mkbond()           # extra open bonds
            A = FT(rand_even([ao, bond], gr), [ao, bond], [false, false], gr)  # A holds bond as out
            B = FT(rand_even([bond, bo], gr), [bond, bo], [true, false], gr)   # B holds bond as in
            AB = fermionic_contract(A, B)
            BA = fermionic_contract(B, A)
            # AB has leg order [ao, bo] and BA has [bo, ao]; as fermionic tensors
            # they are equal only after matching leg order (the swap rule, Eq. 9).
            @test AB.tensor ≈ fermionic_transpose(BA, AB.order).tensor
        end
    end

    @testset "bosonic limit: all-even grading reduces to plain contraction" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mk(d) = (i = Index(d, "x"); set!(gr, i, fill(false, d)); i)
        a, b, c = mk(2), mk(3), mk(2)
        A = FT(ITensor(randn(2, 3), a, b), [a, b], [false, true], gr)
        B = FT(ITensor(randn(3, 2), b, c), [b, c], [false, true], gr)
        C = fermionic_contract(A, B)
        @test C.tensor ≈ A.tensor * B.tensor
    end

    @testset "loopy network: contraction-order independence (triangle)" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mkbond() = (i = Index(2, "b"); set!(gr, i, [false, true]); i)
        bab, bbc, bca = mkbond(), mkbond(), mkbond()
        # arrows: a→b, b→c, c→a (each holds the outgoing bond as out)
        A = FT(rand_even([bab, bca], gr), [bab, bca], [false, true], gr)
        B = FT(rand_even([bab, bbc], gr), [bab, bbc], [true, false], gr)
        C = FT(rand_even([bbc, bca], gr), [bbc, bca], [true, false], gr)

        s1 = fscalar(fermionic_contract(fermionic_contract(A, B), C))
        s2 = fscalar(fermionic_contract(fermionic_contract(B, C), A))
        s3 = fscalar(fermionic_contract(fermionic_contract(C, A), B))
        s4 = fscalar(fermionic_contract(A, fermionic_contract(B, C)))
        @test s1 ≈ s2 ≈ s3 ≈ s4
        @test abs(s1) > 1e-8   # non-trivial
    end

    @testset "loopy network from FTNS with even site caps (square)" begin
        g = named_grid((2, 2))
        s = siteinds("fermion", g)
        ψ = random_fermionic_tensornetworkstate(g, s; bond_dimension = 2)
        gr = TN.grading(ψ)

        # cap each site leg with a random EVEN rank-1 vector (only |0⟩ component),
        # folding it into the vertex tensor so only bond legs remain.
        capped = Dict()
        for v in vertices(ψ)
            ft = FT(ψ, v)
            for si in siteinds(ψ, v)
                vec = zeros(ComplexF64, dim(si)); vec[1] = randn(ComplexF64)  # even component
                cap = FT(ITensor(vec, si), [si], [true], gr)
                ft = fermionic_contract(ft, cap)
            end
            capped[v] = ft
        end

        vs = collect(vertices(ψ))
        orderings = [vs, reverse(vs), shuffle(copy(vs)), shuffle(copy(vs))]
        scalars = ComplexF64[]
        for ord in orderings
            acc = capped[ord[1]]
            for k in 2:length(ord)
                acc = fermionic_contract(acc, capped[ord[k]])
            end
            push!(scalars, fscalar(acc))
        end
        @test all(z -> z ≈ scalars[1], scalars)
    end

    # -----------------------------------------------------------------------
    # Jordan-Wigner ED reference
    #
    # Contract the KET only (independent of the doubled-network code). The open
    # legs are the site indices in some order `modes`; the component array read
    # in that order is the statevector in the basis |n⟩ = ∏_k (c_k†)^{n_k}|0⟩
    # with modes ordered as `modes`. We then build textbook dense JW operators
    # on that same mode order and compare norm_sqr / ⟨N⟩ / ⟨c_i†c_j⟩.
    # -----------------------------------------------------------------------

    # fold-contract all ket tensors -> FermionicTensor whose open legs are the sites
    function ket_ft(ψ)
        vs = collect(vertices(ψ))
        acc = FT(ψ, vs[1])
        for k in 2:length(vs)
            acc = fermionic_contract(acc, FT(ψ, vs[k]))
        end
        return acc
    end

    # dense JW c†_k, c_k, n_k on N modes (column-major basis matching vec(array))
    function jw_ops(N)
        dims = ntuple(_ -> 2, N)
        lin = LinearIndices(dims)
        D = 2^N
        cdag = [zeros(ComplexF64, D, D) for _ in 1:N]
        cann = [zeros(ComplexF64, D, D) for _ in 1:N]
        num = [zeros(ComplexF64, D, D) for _ in 1:N]
        for I in CartesianIndices(dims)
            occ = [I[k] - 1 for k in 1:N]
            col = lin[I]
            for k in 1:N
                sgn = iseven(sum(occ[1:(k - 1)])) ? 1.0 : -1.0
                num[k][col, col] = occ[k]
                newocc = copy(occ)
                if occ[k] == 0
                    newocc[k] = 1
                    row = lin[CartesianIndex(ntuple(d -> newocc[d] + 1, N))]
                    cdag[k][row, col] = sgn
                else
                    newocc[k] = 0
                    row = lin[CartesianIndex(ntuple(d -> newocc[d] + 1, N))]
                    cann[k][row, col] = sgn
                end
            end
        end
        return cdag, cann, num
    end

    @testset "expect vs Jordan-Wigner ED ($name)" for (name, g) in
        ("chain" => named_grid((4, 1)), "grid2x2" => named_grid((2, 2)), "grid2x3" => named_grid((2, 3)))
        Random.seed!(2468)
        s = siteinds("fermion", g)
        ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)

        K = ket_ft(ψ)
        modes = K.order                                  # site indices in leg order
        N = length(modes)
        ψvec = vec(ITensors.array(K.tensor, modes...))
        cdag, cann, num = jw_ops(N)

        vs = collect(vertices(ψ))
        pos = Dict(v => findfirst(==(only(siteinds(ψ, v))), modes) for v in vs)

        nrm = real(ψvec' * ψvec)
        @test norm_sqr(ψ) ≈ nrm

        # ⟨N_v⟩ (even operator)
        for v in vs
            ed = (ψvec' * (num[pos[v]] * ψvec)) / nrm
            @test expect(ψ, ("N", [v])) ≈ ed
        end

        # ⟨c_i† c_j⟩ hopping (odd pair -> string dummy bond)
        for i in vs, j in vs
            i == j && continue
            ed = (ψvec' * (cdag[pos[i]] * (cann[pos[j]] * ψvec))) / nrm
            @test expect(ψ, (["Cdag", "C"], [i, j])) ≈ ed
        end
    end
end
end
