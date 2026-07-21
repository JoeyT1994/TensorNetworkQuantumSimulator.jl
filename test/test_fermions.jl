@eval module $(gensym())
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws
using Random
using ITensors
using ITensors: ITensors, Index, ITensor, dim, inds, contract, permute, scalar
using Dictionaries: Dictionary, set!
using Graphs: ne
using SparseArrays: sparse, SparseMatrixCSC
using TensorNetworkQuantumSimulator: random_even_itensor

const FT = TN.FermionicITensor

ITensors.disable_warn_order()

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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

    @testset "permute vs brute oracle" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mk(d, bits) = (i = Index(d, "x"); set!(gr, i, bits); i)
        for _ in 1:20
            is = [mk(2, [false, true]), mk(3, [false, true, true]), mk(2, [false, true]), mk(2, [true, false])]
            T = random_even_itensor(is, gr)
            dirs = rand(Bool, length(is))
            ft = FT(T, copy(is), dirs, gr)
            to = shuffle(is)
            fast = permute(ft, to)
            brute = brute_reorder(gr, T, is, to)
            @test fast.tensor ≈ brute
            @test fast.order == to
            # dirs travel with their legs
            @test all(fast.dirs[k] == dirs[findfirst(==(to[k]), is)] for k in 1:length(is))
        end
    end

    @testset "dag is an involution" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mk(d, bits) = (i = Index(d, "x"); set!(gr, i, bits); i)
        is = [mk(2, [false, true]), mk(2, [false, true]), mk(3, [false, true, true])]
        T = random_even_itensor(is, gr)
        ft = FT(T, copy(is), rand(Bool, 3), gr)
        dd = dag(dag(ft))
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
            A = FT(random_even_itensor([ao, bond], gr), [ao, bond], [false, false], gr)  # A holds bond as out
            B = FT(random_even_itensor([bond, bo], gr), [bond, bo], [true, false], gr)   # B holds bond as in
            AB = contract(A, B)
            BA = contract(B, A)
            # AB has leg order [ao, bo] and BA has [bo, ao]; as fermionic tensors
            # they are equal only after matching leg order (the swap rule, Eq. 9).
            @test AB.tensor ≈ permute(BA, AB.order).tensor
        end
    end

    @testset "bosonic limit: all-even grading reduces to plain contraction" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mk(d) = (i = Index(d, "x"); set!(gr, i, fill(false, d)); i)
        a, b, c = mk(2), mk(3), mk(2)
        A = FT(ITensor(randn(2, 3), a, b), [a, b], [false, true], gr)
        B = FT(ITensor(randn(3, 2), b, c), [b, c], [false, true], gr)
        C = contract(A, B)
        @test C.tensor ≈ A.tensor * B.tensor
    end

    @testset "loopy network: contraction-order independence (triangle)" begin
        gr = Dictionary{Index, Vector{Bool}}()
        mkbond() = (i = Index(2, "b"); set!(gr, i, [false, true]); i)
        bab, bbc, bca = mkbond(), mkbond(), mkbond()
        # arrows: a→b, b→c, c→a (each holds the outgoing bond as out)
        A = FT(random_even_itensor([bab, bca], gr), [bab, bca], [false, true], gr)
        B = FT(random_even_itensor([bab, bbc], gr), [bab, bbc], [true, false], gr)
        C = FT(random_even_itensor([bbc, bca], gr), [bbc, bca], [true, false], gr)

        s1 = scalar(contract(contract(A, B), C))
        s2 = scalar(contract(contract(B, C), A))
        s3 = scalar(contract(contract(C, A), B))
        s4 = scalar(contract(A, contract(B, C)))
        @test s1 ≈ s2 ≈ s3 ≈ s4
        @test abs(s1) > 1e-8   # non-trivial
    end

    @testset "cached contraction sign masks" begin
        function triangle_fermionic_tensors(bits = [false, true])
            gr = Dictionary{Index, Vector{Bool}}()
            mkbond() = (i = Index(2, "b"); set!(gr, i, copy(bits)); i)
            bab, bbc, bca = mkbond(), mkbond(), mkbond()
            A = FT(
                random_even_itensor([bab, bca], gr),
                [bab, bca],
                [false, true],
                gr,
            )
            B = FT(
                random_even_itensor([bab, bbc], gr),
                [bab, bbc],
                [true, false],
                gr,
            )
            C = FT(
                random_even_itensor([bbc, bca], gr),
                [bbc, bca],
                [true, false],
                gr,
            )
            return [A, B, C]
        end

        sequence = [[1, 2], 3]
        sign_plans = TN.FermionicBinaryContractionPlan[]

        fts = triangle_fermionic_tensors()
        cached = contract(fts; sequence, sign_plans)
        @test cached.tensor ≈ contract(fts; sequence).tensor
        @test length(sign_plans) == 2
        @test any(
            p -> (!isnothing(p.plan_a) && !isnothing(p.plan_a.mask)) ||
                (!isnothing(p.plan_b) && !isnothing(p.plan_b.mask)),
            sign_plans,
        )

        old_signatures = [
            (isnothing(p.plan_a) ? nothing : p.plan_a.signature,
             isnothing(p.plan_b) ? nothing : p.plan_b.signature) for p in sign_plans
        ]
        old_masks = [
            (isnothing(p.plan_a) ? nothing : p.plan_a.mask,
             isnothing(p.plan_b) ? nothing : p.plan_b.mask) for p in sign_plans
        ]

        # Untruncated QR creates fresh Index identities but preserves this structural
        # information, so an equivalent fresh network should reuse the same mask objects.
        fresh_fts = triangle_fermionic_tensors()
        fresh_cached = contract(fresh_fts; sequence, sign_plans)
        @test fresh_cached.tensor ≈ contract(fresh_fts; sequence).tensor
        @test all(
            (isnothing(p.plan_a) ? nothing : p.plan_a.mask) === masks[1] &&
                (isnothing(p.plan_b) ? nothing : p.plan_b.mask) === masks[2]
            for (p, masks) in zip(sign_plans, old_masks)
        )

        # A grading change with identical dimensions and leg orders must invalidate
        # the plans rather than silently applying masks for the old grading.
        changed_fts = triangle_fermionic_tensors([true, false])
        changed_cached = contract(changed_fts; sequence, sign_plans)
        @test changed_cached.tensor ≈ contract(changed_fts; sequence).tensor
        @test any(
            (isnothing(p.plan_a) ? nothing : p.plan_a.signature,
             isnothing(p.plan_b) ? nothing : p.plan_b.signature) != signatures
            for (p, signatures) in zip(sign_plans, old_signatures)
        )
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
                ft = contract(ft, cap)
            end
            capped[v] = ft
        end

        vs = collect(vertices(ψ))
        orderings = [vs, reverse(vs), shuffle(copy(vs)), shuffle(copy(vs))]
        scalars = ComplexF64[]
        for ord in orderings
            acc = capped[ord[1]]
            for k in 2:length(ord)
                acc = contract(acc, capped[ord[k]])
            end
            push!(scalars, scalar(acc))
        end
        @test all(z -> z ≈ scalars[1], scalars)
    end

    @testset "fermionic loop corrections reproduce exact Z ($name)" for (name, g) in
        ("grid2x2" => named_grid((2, 2)),   # single chordless plaquette
         "grid2x3" => named_grid((2, 3)),   # contains a chorded 6-cycle
         "grid3x3" => named_grid((3, 3)))   # chorded cycles + disconnected configs
        Random.seed!(42)
        s = siteinds("fermion", g)
        ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)
        zexact = norm_sqr(ψ; alg = "exact")
        cuk = (; maxiter = 200, tol = 1e-14)
        # Including every no-leaf edge-induced subgraph (the full cluster series, via the
        # 𝟙 − P fermionic antiprojector with chords capped by the BP projector) is an exact
        # resummation of Z at the BP fixed point.
        zloop = norm_sqr(ψ; alg = "loopcorrections", max_configuration_size = ne(g),
                         cache_update_kwargs = cuk)
        @test zloop ≈ zexact rtol = 1e-8
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

    # fold-contract all ket tensors -> FermionicITensor whose open legs are the sites
    function ket_ft(ψ)
        vs = collect(vertices(ψ))
        acc = ψ[vs[1]]
        for k in 2:length(vs)
            acc = contract(acc, ψ[vs[k]])
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
        ("chain" => named_grid((4, 1)), "comb3x3" => named_comb_tree((3,3)), "grid2x2" => named_grid((2, 2)), "grid2x3" => named_grid((2, 3)))
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
        @test norm_sqr(ψ; alg = "exact") ≈ nrm
        if is_tree(g)
            @test norm_sqr(ψ; alg = "bp") ≈ nrm
        end

        # ⟨N_v⟩ (even operator)
        for v in vs
            ed = (ψvec' * (num[pos[v]] * ψvec)) / nrm
            @test expect(ψ, ("N", [v]); alg = "exact") ≈ ed
            if is_tree(g)
                @test expect(ψ, ("N", [v]); alg = "exact") ≈ expect(ψ, ("N", [v]); alg = "bp")
            end
        end

        # ⟨c_i† c_j⟩ hopping (odd pair -> string dummy bond)
        for i in vs, j in vs
            i == j && continue
            ed = (ψvec' * (cdag[pos[i]] * (cann[pos[j]] * ψvec))) / nrm
            @test expect(ψ, (["Cdag", "C"], [i, j]); alg = "exact") ≈ ed
            if is_tree(g)
                @test expect(ψ, (["Cdag", "C"], [i, j]); alg = "exact") ≈ expect(ψ, (["Cdag", "C"], [i, j]); alg = "bp")
            end
        end

        # single odd operator: parity-forbidden ⇒ ⟨O⟩ = 0 (no error).
        # multi-character op names must use the vector form (a bare String is read
        # as one operator character per vertex).
        for v in vs
            @test expect(ψ, (["Cdag"], [v]); alg = "exact") ≈ 0 atol = 1e-12
            @test expect(ψ, (["C"], [v]); alg = "exact") ≈ 0 atol = 1e-12
        end
    end

    # ---------------------------------------------------------------------------
    # SPINFUL expectation values vs an independent Jordan-Wigner statevector reference.
    #
    # This is the spinful analogue of the spinless "expect vs JW ED" testset above. It
    # specifically stresses the on-site opposite-spin Jordan-Wigner string: cross-site
    # bilinears ⟨c†↑_i c↑_j⟩, ⟨c†↓_i c↓_j⟩ and the spin-flip ⟨c†↑_i c↓_j⟩ on a generic
    # entangled state that carries weight in every (n↑,n↓) sector, including double
    # occupancy. The expectation code path threads the operator string through a single
    # dim-1 dummy bond `d` via `odd_op_tensor` — exactly the mechanism whose hopping
    # analogue dropped the spectator opposite-spin sign on double-occupancy transitions.
    #
    # Reference: the dim-4-per-site Fock operators (AUP/ADN with their intra-site ↑-before-↓
    # sign baked in) with PAR=(−1)^{n↑+n↓} strings between sites — the SAME construction
    # proven exactly equivalent to the spinless jw_ops(2N) reference in the Hubbard test
    # below. A single number operator is built as embed(i,·dag)*embed(i,·) (the j<i strings
    # square to the identity), so the same `embed` serves both even and odd observables.
    # ---------------------------------------------------------------------------
    @testset "spinful expect vs JW statevector ($name)" for (name, g) in
        ("chain3" => named_grid((3, 1)), "grid2x2" => named_grid((2, 2)), "grid2x3" => named_grid((2, 3)))
        Random.seed!(13579)
        AUP = sparse(ComplexF64[0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0])
        ADN = sparse(ComplexF64[0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0])
        AUPDAG = sparse(collect(AUP')); ADNDAG = sparse(collect(ADN'))
        PAR = sparse(ComplexF64[1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1])
        ID4 = sparse(ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
        # operator on site k (1 = fastest mode) with parity strings on every site j < k
        function embed(N, k, A)
            mats = [j < k ? PAR : (j == k ? A : ID4) for j in 1:N]
            op = mats[N]; for j in (N - 1):-1:1; op = kron(op, mats[j]); end; return op
        end

        s = siteinds("spinful_fermion", g)
        ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)

        K = ket_ft(ψ)
        modes = K.order                                  # site indices in leg order
        N = length(modes)
        ψvec = vec(ITensors.array(K.tensor, modes...))
        nrm = real(ψvec' * ψvec)
        @test norm_sqr(ψ; alg = "exact") ≈ nrm

        vs = collect(vertices(ψ))
        pos = Dict(v => findfirst(==(only(siteinds(ψ, v))), modes) for v in vs)

        # even single-site number operators ⟨N↑_i⟩, ⟨N↓_i⟩
        for v in vs
            k = pos[v]
            edup = (ψvec' * (embed(N, k, AUPDAG) * (embed(N, k, AUP) * ψvec))) / nrm
            eddn = (ψvec' * (embed(N, k, ADNDAG) * (embed(N, k, ADN) * ψvec))) / nrm
            @test expect(ψ, (["Nup"], [v]); alg = "exact") ≈ edup
            @test expect(ψ, (["Ndn"], [v]); alg = "exact") ≈ eddn
        end

        # cross-site odd bilinears — the on-site opposite-spin string is exercised here
        for i in vs, j in vs
            i == j && continue
            ki, kj = pos[i], pos[j]
            for (opd, opa, Ad, Aa) in (
                    ("Cupdag", "Cup", AUPDAG, AUP),     # ⟨c†↑_i c↑_j⟩
                    ("Cdndag", "Cdn", ADNDAG, ADN),     # ⟨c†↓_i c↓_j⟩
                    ("Cupdag", "Cdn", AUPDAG, ADN),     # ⟨c†↑_i c↓_j⟩ (spin flip)
                    ("Cdndag", "Cup", ADNDAG, AUP),     # ⟨c†↓_i c↑_j⟩
                )
                ed = (ψvec' * (embed(N, ki, Ad) * (embed(N, kj, Aa) * ψvec))) / nrm
                @test expect(ψ, ([opd, opa], [i, j]); alg = "exact") ≈ ed
            end
        end

        # single odd operator: parity-forbidden ⇒ ⟨O⟩ = 0
        for v in vs
            @test expect(ψ, (["Cupdag"], [v]); alg = "exact") ≈ 0 atol = 1e-12
            @test expect(ψ, (["Cdn"], [v]); alg = "exact") ≈ 0 atol = 1e-12
        end
    end

    # ---------------------------------------------------------------------------
    # product-state constructor (fermionic_tensornetworkstate)
    # ---------------------------------------------------------------------------
    @testset "product state: spinless ($name)" for (name, g) in
        ("chain" => named_grid((4, 1)), "grid2x2" => named_grid((2, 2)))
        s = siteinds("fermion", g)
        vs = collect(vertices(g))
        # occupy every other vertex; flip one if needed so the total parity is even
        occ = Dictionary(vs, [isodd(i) ? 1 : 0 for i in eachindex(vs)])
        isodd(sum(occ)) && (occ[first(vs)] = 1 - occ[first(vs)])

        ψ = fermionic_tensornetworkstate(v -> occ[v] == 1 ? "Occ" : "Emp", g, s)
        @test norm_sqr(ψ; alg = "exact") ≈ 1
        is_tree(g) && @test norm_sqr(ψ; alg = "bp") ≈ 1
        for v in vs
            @test expect(ψ, (["N"], [v]); alg = "exact") ≈ occ[v]
            is_tree(g) && @test expect(ψ, (["N"], [v]); alg = "bp") ≈ occ[v]
        end
    end

    @testset "product state: spinful" begin
        g = named_grid((2, 2))
        s = siteinds("spinful_fermion", g)
        vs = collect(vertices(g))
        # Up (odd), Dn (odd), UpDn (even), Emp (even) -> total parity even
        names = Dictionary(vs, ["Up", "Dn", "UpDn", "Emp"])
        nup = Dictionary(vs, [1, 0, 1, 0])
        ndn = Dictionary(vs, [0, 1, 1, 0])

        ψ = fermionic_tensornetworkstate(v -> names[v], g, s)
        @test norm_sqr(ψ; alg = "exact") ≈ 1
        for v in vs
            @test expect(ψ, (["Nup"], [v]); alg = "exact") ≈ nup[v]
            @test expect(ψ, (["Ndn"], [v]); alg = "exact") ≈ ndn[v]
        end
    end

    @testset "product state: vector-form local states" begin
        g = named_grid((2, 1))
        s = siteinds("fermion", g)
        # |1> on both sites (parity-definite vectors); 2 fermions -> even total
        ψ = fermionic_tensornetworkstate(v -> [0.0, 1.0], g, s)
        @test norm_sqr(ψ; alg = "exact") ≈ 1
        for v in vertices(g)
            @test expect(ψ, (["N"], [v]); alg = "exact") ≈ 1
        end
    end

    @testset "product state: eltype + convenience dispatch" begin
        g = named_grid((2, 1))
        s = siteinds("fermion", g)
        ψ = fermionic_tensornetworkstate(ComplexF64, v -> "Emp", g, s)
        @test eltype(FT(ψ, first(vertices(g))).tensor) == ComplexF64
    end

    @testset "product state: errors" begin
        g = named_grid((3, 1))
        s = siteinds("fermion", g)
        # odd total fermion parity (a single fermion) is not representable
        @test_throws ErrorException fermionic_tensornetworkstate(
            v -> v == first(vertices(g)) ? "Occ" : "Emp", g, s)
        # coherent parity superposition (|0> + |1>) is forbidden
        @test_throws ErrorException fermionic_tensornetworkstate(v -> ComplexF64[1, 1], g, s)
    end
end
end
