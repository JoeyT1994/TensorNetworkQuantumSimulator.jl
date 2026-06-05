@eval module $(gensym())
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws
using Random
using ITensors
using ITensors: ITensors, Index, ITensor, dim, inds, contract, permute, scalar
using Dictionaries: Dictionary, set!
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
    # simple_update of a two-site gate vs Jordan-Wigner ED
    #
    # Apply the real-time hopping gate exp(-i dt H_hop) on one edge through the full
    # BP-gauge `simple_update` path and compare the reconstructed ket to the exact JW
    # propagator on the full mode space. A third site makes the environment/QR bond carry
    # odd parity, which is exactly where a naive fermionic-blob (or sign-missing) gate
    # application corrupts the odd-odd hopping channel. No truncation ⇒ must be exact.
    # ---------------------------------------------------------------------------
    @testset "simple_update vs Jordan-Wigner ED ($name)" for (name, g) in
        ("chain3" => named_grid((3, 1)), "chain4" => named_grid((4, 1)))
        Random.seed!(2468)
        l2(x) = sqrt(real(sum(abs2, x)))
        s = siteinds("fermion", g)
        ψ = TN.random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)
        vs = collect(vertices(ψ))

        K = ket_ft(ψ)
        modes = K.order
        N = length(modes)
        ψvec = vec(ITensors.array(K.tensor, modes...))
        cdag, cann, num = jw_ops(N)
        Ntot = sum(num)

        # apply the gate on the first edge (v1,v2); remaining sites are spectators
        v1, v2 = vs[1], vs[2]
        s1 = only(siteinds(ψ, v1)); s2 = only(siteinds(ψ, v2))
        p1 = findfirst(==(s1), modes); p2 = findfirst(==(s2), modes)
        dt = 0.37
        Hjw = cdag[p1] * cann[p2] + cdag[p2] * cann[p1]
        target = exp(-im * dt * Hjw) * ψvec

        o = TN.fermionic_hopping_gate(dt, s1, s2)
        bpc = TN.update(TN.BeliefPropagationCache(ψ))
        envs = TN.incoming_messages(bpc, [v1, v2])
        updated, _, err = TN.simple_update(
            o, TN.network(bpc), [v1, v2]; envs, normalize_tensors = false, maxdim = 64,
        )

        # reconstruct the full ket from the two updated tensors + untouched spectators
        new_tensors = Dictionary(vs, [k == 1 ? updated[1] : k == 2 ? updated[2] : ψ[vs[k]] for k in eachindex(vs)])
        acc = new_tensors[vs[1]]
        for k in 2:length(vs)
            acc = contract(acc, new_tensors[vs[k]])
        end
        newvec = vec(ITensors.array(noprime(acc).tensor, modes...))

        @test err ≈ 0 atol = 1e-12                              # no truncation
        @test l2(newvec - target) ≈ 0 atol = 1e-10              # matches JW exactly
        # total fermion number is conserved by hopping
        nb = real(ψvec' * Ntot * ψvec) / real(ψvec' * ψvec)
        na = real(newvec' * Ntot * newvec) / real(newvec' * newvec)
        @test na ≈ nb atol = 1e-10
    end

    # ---------------------------------------------------------------------------
    # On-site Hubbard interaction gate exp(coeff·dt·n↑n↓) on spinful sites. The gate is
    # single-site and diagonal (n↑n↓ is the double-occupancy projector), so its dense array
    # must be diag(1,1,1,exp(coeff·dt)) and applying it must equal multiplying the folded ket
    # by that same local diagonal operator (basis state |↑↓⟩ = index 4 of the 4-dim leg).
    # ---------------------------------------------------------------------------
    @testset "interaction gate exp(coeff·dt·n↑n↓) (coeff=$coeff)" for coeff in (-im, -1.0)
        l2(x) = sqrt(real(sum(abs2, x)))
        dt = 0.41

        # (a) direct matrix check on an isolated spinful site
        s0 = only(siteinds("spinful_fermion", named_grid((1, 1)))[(1, 1)])
        o0 = TN.fermionic_interaction_gate(dt, s0; coeff)
        Mmat = reshape(Array(ITensors.array(o0.tensor, prime(s0), s0)), 4, 4)
        expected = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(coeff * dt)]
        @test l2(Mmat - expected) ≈ 0 atol = 1e-12

        # (b) apply via simple_update on a spinful chain; compare to the local diagonal op
        Random.seed!(99)
        g = named_grid((3, 1))
        ss = siteinds("spinful_fermion", g)
        ψ = random_fermionic_tensornetworkstate(ComplexF64, g, ss; bond_dimension = 2)
        vs = collect(vertices(ψ))
        K = ket_ft(ψ); modes = K.order; N = length(modes)
        ψvec = vec(ITensors.array(K.tensor, modes...))

        v1 = vs[1]; s1 = only(siteinds(ψ, v1))
        p1 = findfirst(==(s1), modes)
        o = TN.fermionic_interaction_gate(dt, s1; coeff)

        # local diagonal operator on the full (4^N) Hilbert space: phase only when site 1 is
        # doubly occupied (local basis index 4 ⇒ |↑↓⟩)
        Dvec = ones(ComplexF64, 4^N)
        for (lin, I) in enumerate(CartesianIndices(ntuple(_ -> 4, N)))
            I[p1] == 4 && (Dvec[lin] *= exp(coeff * dt))
        end
        target = Dvec .* ψvec

        updated, _, err = TN.simple_update(o, ψ, [v1]; envs = FT[], normalize_tensors = false)
        new_tensors = [k == 1 ? updated[1] : ψ[vs[k]] for k in eachindex(vs)]
        acc = new_tensors[1]
        for k in 2:length(vs); acc = contract(acc, new_tensors[k]); end
        newvec = vec(ITensors.array(noprime(acc).tensor, modes...))
        @test err ≈ 0 atol = 1e-12
        @test l2(newvec - target) ≈ 0 atol = 1e-10
    end

    # ---------------------------------------------------------------------------
    # End-to-end Hubbard real-time fidelity vs bond dimension on a 3×2 grid.
    #
    # Run a short 2nd-order Trotter circuit for H = -t Σ_⟨ij⟩σ (c†_iσ c_jσ + h.c.)
    #   + U Σ_i n_i↑ n_i↓  via the BP-gauge `apply_gate!` path at increasing maxdim,
    # contract the whole network down to a single ket, and compare to an INDEPENDENT
    # dense Jordan-Wigner statevector evolved by the same circuit. The reference uses
    # textbook sparse JW matrices (per-site parity string P = diag(1,−1,−1,1) threaded
    # between sites; site 1 the fastest index) and a Taylor matrix-exponential — a
    # completely different code path from the locally-ordered fermionic tensors. As the
    # bond dimension grows the truncation error vanishes, so the fidelity must increase
    # monotonically toward 1 (a 6-site, 2-step circuit is exactly representable). A
    # systematic sign error in any gate (e.g. a dropped Jordan-Wigner string on a
    # double-occupancy hop) would instead saturate the fidelity well below 1.
    # ---------------------------------------------------------------------------
    @testset "Hubbard 3×2 real-time: fidelity vs bond dimension (JW statevector ref)" begin
        l2v(x) = sqrt(real(sum(abs2, x)))

        # textbook spinful JW local matrices (basis |0⟩,|↑⟩,|↓⟩,|↑↓⟩); intra-site ↑-before-↓
        # ordering is baked into the −1 in ADN, P = (−1)^{n↑+n↓} is the inter-site string.
        AUP = sparse(ComplexF64[0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0])
        ADN = sparse(ComplexF64[0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0])
        AUPDAG = sparse(collect(AUP'))
        ADNDAG = sparse(collect(ADN'))
        PAR = sparse(ComplexF64[1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1])
        ID4 = sparse(ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])

        # embed a local 4×4 op at mode position `k` with JW parity strings on earlier
        # (faster) modes. Site 1 is the fastest index ⇒ kron(mat_N, …, mat_1).
        function embed(N, k, A)
            mats = [j < k ? PAR : (j == k ? A : ID4) for j in 1:N]
            op = mats[N]
            for j in (N - 1):-1:1
                op = kron(op, mats[j])
            end
            return op
        end
        # exp(θ·H)·v by Taylor; θ‖H‖ ≪ 1 here so this is machine-exact.
        function applyexp(H, v, θ; K = 40)
            term = v; acc = copy(v)
            for n in 1:K
                term = (θ / n) * (H * term)
                acc += term
                l2v(term) < 1e-17 * l2v(acc) && break
            end
            return acc
        end

        dims = (3, 2); nsteps = 2; dt = 0.1; U = 5.0; t = -1.0
        chis = [1, 2, 4, 8]
        g = named_grid(dims)
        s = siteinds("spinful_fermion", g)
        vs = collect(vertices(g))
        namef = v -> isodd(sum(v)) ? "Up" : "Dn"
        foldket(ψ) = (acc = ψ[vs[1]]; for k in 2:length(vs); acc = contract(acc, ψ[vs[k]]); end; acc)

        # fix the mode order from the initial product state
        ψ0 = fermionic_tensornetworkstate(ComplexF64, namef, g, s)
        K0 = foldket(ψ0)
        modes = K0.order
        N = length(modes)
        pos = Dict(only(s[v]) => findfirst(==(only(s[v])), modes) for v in vs)

        # independent JW reference: start in the same product state
        nameidx = Dict("Up" => 2, "Dn" => 3, "Emp" => 1, "UpDn" => 4)
        lin0 = 1
        for v in vs
            lin0 += (nameidx[namef(v)] - 1) * 4^(pos[only(s[v])] - 1)
        end
        vref = zeros(ComplexF64, 4^N); vref[lin0] = 1

        ec = edge_color(g, 4)
        # interaction half-step diagonal per site position
        int_half = Dict{Int, Vector{ComplexF64}}()
        for v in vs
            k = pos[only(s[v])]
            full = ones(ComplexF64, 4^N)
            for (lin, I) in enumerate(CartesianIndices(ntuple(_ -> 4, N)))
                I[k] == 4 && (full[lin] *= exp(-0.5im * U * dt))
            end
            int_half[k] = full
        end
        # hopping H per edge
        Hbond = Dict{Tuple{Int, Int}, SparseMatrixCSC{ComplexF64, Int}}()
        for es in ec, e in es
            a = pos[only(s[src(e)])]; b = pos[only(s[dst(e)])]
            Hbond[(a, b)] = embed(N, a, AUPDAG) * embed(N, b, AUP) + embed(N, b, AUPDAG) * embed(N, a, AUP) +
                embed(N, a, ADNDAG) * embed(N, b, ADN) + embed(N, b, ADNDAG) * embed(N, a, ADN)
        end
        θhop = (-t * im) * dt
        for _ in 1:nsteps
            for v in vs; vref = int_half[pos[only(s[v])]] .* vref; end
            for es in ec, e in es
                a = pos[only(s[src(e)])]; b = pos[only(s[dst(e)])]
                vref = applyexp(Hbond[(a, b)], vref, θhop)
            end
            for v in vs; vref = int_half[pos[only(s[v])]] .* vref; end
        end
        vref ./= l2v(vref)

        # run the TNS circuit at each bond dimension, fold to a ket, measure fidelity
        single_gates = [fermionic_interaction_gate(dt, only(s[v]); coeff = -0.5im * U) for v in vs]
        hop_groups = [[fermionic_hopping_gate(dt, only(s[src(e)]), only(s[dst(e)]); coeff = -t * im) for e in es] for es in ec]

        fids = Float64[]
        for χ in chis
            ψ_bpc = update(BeliefPropagationCache(fermionic_tensornetworkstate(ComplexF64, namef, g, s)))
            rescale!(ψ_bpc)
            apply_kwargs = (; maxdim = χ, cutoff = 1e-16)
            for _ in 1:nsteps
                for gate in single_gates; TN.apply_gate!(gate, ψ_bpc; apply_kwargs); end
                for grp in hop_groups, gate in grp; TN.apply_gate!(gate, ψ_bpc; apply_kwargs); end
                for gate in single_gates; TN.apply_gate!(gate, ψ_bpc; apply_kwargs); end
                ψ_bpc = update(ψ_bpc)
            end
            Kf = foldket(network(ψ_bpc))
            @test Kf.order == modes                       # mode order stable across χ
            kv = vec(ITensors.array(Kf.tensor, modes...)); kv ./= l2v(kv)
            push!(fids, abs(sum(conj.(vref) .* kv)))
        end

        # the low-χ state is genuinely truncated (discriminating), and the fidelity
        # converges monotonically to 1 as the bond dimension grows.
        @test fids[1] < 0.95
        @test all(fids[k] ≤ fids[k + 1] + 1e-6 for k in 1:(length(fids) - 1))
        @test fids[end] > 0.999
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
