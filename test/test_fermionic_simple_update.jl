@eval module $(gensym())
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test
using Random
using ITensors
using ITensors: ITensors, Index, ITensor, dim, inds, contract, permute, scalar
using Dictionaries: Dictionary, set!
using SparseArrays: sparse, SparseMatrixCSC

const FT = TN.FermionicITensor

ITensors.disable_warn_order()

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# fold a fermionic TNS down to a single FermionicITensor ket
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

@testset "Test fermionic simple_update" begin
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

        # run the TNS circuit at each bond dimension, fold to a ket, measure fidelity.
        # First Trotter step: gates built manually and applied via `apply_gate!`.
        single_gates = [fermionic_interaction_gate(dt, only(s[v]); coeff = -0.5im * U) for v in vs]
        hop_groups = [[fermionic_hopping_gate(dt, only(s[src(e)]), only(s[dst(e)]); coeff = -t * im) for e in es] for es in ec]
        # Remaining Trotter steps: the same half-step layout as a tuple circuit parsed
        # by `apply_gates`. `RInt` bakes in the `-0.5·im` half-step exponent, so its angle
        # is θ_int = U·dt (a multiplier on `-0.5·im`); `RHop` rotates by θ_hop = t·dt.
        step_circuit = vcat(
            [("RInt", [v], U * dt) for v in vs],
            [("RHop", [src(e), dst(e)], t * dt) for es in ec for e in es],
            [("RInt", [v], U * dt) for v in vs],
        )

        fids = Float64[]
        for χ in chis
            ψ_bpc = update(BeliefPropagationCache(fermionic_tensornetworkstate(ComplexF64, namef, g, s)))
            rescale!(ψ_bpc)
            apply_kwargs = (; maxdim = χ, cutoff = 1e-16)
            # first step: manual gate list
            for gate in single_gates; TN.apply_gate!(gate, ψ_bpc; apply_kwargs); end
            for grp in hop_groups, gate in grp; TN.apply_gate!(gate, ψ_bpc; apply_kwargs); end
            for gate in single_gates; TN.apply_gate!(gate, ψ_bpc; apply_kwargs); end
            ψ_bpc = update(ψ_bpc)
            # remaining steps: tuple-string circuit through the parser
            # (`apply_gates` updates the BP cache internally at the end)
            for _ in 2:nsteps
                ψ_bpc, _ = apply_gates(step_circuit, ψ_bpc; apply_kwargs)
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
end
end
