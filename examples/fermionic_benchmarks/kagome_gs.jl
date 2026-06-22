using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge
Random.seed!(1234)
using LinearAlgebra

"""
    named_kagome_lattice_graph(nx, ny) -> NamedGraph

Finite, open-boundary Kagome lattice of `nx × ny` triangular Bravais cells. Each cell carries
three sublattice sites and vertices are labelled `(i, j, s)` with `i ∈ 1:nx`, `j ∈ 1:ny`,
`s ∈ 1:3`. With Bravais vectors `a1 = (1,0)`, `a2 = (1/2, √3/2)` the sublattice positions in
cell `(i,j)` are `s=1` at `r`, `s=2` at `r + a1/2`, `s=3` at `r + a2/2` (`r = i·a1 + j·a2`).

The lattice is the corner-sharing-triangle net: each cell contributes an up-triangle
`{1,2,3}` (intra-cell) and the three inter-cell bonds that, with up-triangle edges of
neighbouring cells, close the down-triangles. Inter-cell bonds are added only when the
neighbour cell lies in `1:nx × 1:ny`, so boundary sites have reduced coordination (bulk sites
are 4-coordinate). The patch is a rhombus (because `a2` sits at 60°).
"""
function named_kagome_lattice_graph(nx::Int, ny::Int)
    verts = [(i, j, s) for i in 1:nx for j in 1:ny for s in 1:3]
    g = NamedGraph(verts)
    inrange(i, j) = 1 <= i <= nx && 1 <= j <= ny
    for i in 1:nx, j in 1:ny
        # up-triangle (intra-cell): 1—2—3—1
        g = add_edge!(g, NamedEdge((i, j, 1) => (i, j, 2)))
        g = add_edge!(g, NamedEdge((i, j, 2) => (i, j, 3)))
        g = add_edge!(g, NamedEdge((i, j, 3) => (i, j, 1)))
        # down-triangle (inter-cell), each guarded by an open boundary:
        inrange(i + 1, j)     && (g = add_edge!(g, NamedEdge((i, j, 2) => (i + 1, j, 1))))      # 2 → 1 (+a1)
        inrange(i, j + 1)     && (g = add_edge!(g, NamedEdge((i, j, 3) => (i, j + 1, 1))))      # 3 → 1 (+a2)
        inrange(i + 1, j - 1) && (g = add_edge!(g, NamedEdge((i, j, 2) => (i + 1, j - 1, 3))))  # 2 → 3 (+a1−a2)
    end
    return g
end

# Tang et al. (arXiv:1111.1172, Fig. 21 / Eq. 7): NN spin–orbit Kagome.
# A=1, B=2, C=3. Arrows circulate A→B→C→A on every triangle; along an arrow the
# amplitude is (t1 - im*λ1). Edge (v1,v2) ⇒ t_ij is the coeff of c†_{v1} c_{v2}.
function kagome_amp(v1, v2, t1, λ1)
    s1, s2 = v1[3], v2[3]
    along = (s2, s1) in ((1, 2), (2, 3), (3, 1))   # arrow head sits at v1
    return along ? (t1 - im * λ1) : (t1 + im * λ1)
end


function bp_energy(ψ_bpc::BeliefPropagationCache, t, ϕ)
    g = graph(ψ_bpc)
    e_hop = 0
    t1, λ1  = real(exp(im * ϕ)), imag(exp(im * ϕ))
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        tij = kagome_amp(v1, v2, t1, λ1)
        e_hop += tij * expect(ψ_bpc, (["Cdag", "C"], [v1, v2])) + conj(tij) * expect(ψ_bpc, (["Cdag", "C"], [v2, v1]))
    end

    return t*e_hop
end

function kagome_gates(s, g, dt, ϕ, t_hop)
    # Peierls phase along the DIRECTED bond v1 -> v2. Orient every bond consistently
    # (here src->dst from edges(g)); the directed sum of φ around a triangle/hexagon
    # must equal the flux you want to thread that plaquette.
    hop_gates    = FermionicITensor[]  
    ec = edge_color(g, 4)
    # vector of FermionicITensor
    t1, λ1  = real(exp(im * ϕ)), imag(exp(im * ϕ))
    for es in ec
        for e in es
            v1, v2 = src(e), dst(e)
            t_ij = kagome_amp(v1, v2, t1, λ1)
            push!(hop_gates, fermionic_hopping_gate(dt, only(s[v1]), only(s[v2]); t = t_ij, coeff = -1))
        end
    end
    return hop_gates
end

"""
    free_fermion_gs_energy(g, t, ϕ, n_fermions) -> (E_gs, ε)

Exact GS energy of the quadratic hopping Hamiltonian that `bp_energy` measures,
at fixed particle number `n_fermions`. Builds the N×N single-particle matrix
`h` (h[a,b] = coefficient of c†_a c_b), diagonalizes, and sums the `n_fermions`
lowest eigenvalues. Also returns the full ascending spectrum `ε`.
"""
function free_fermion_gs_energy(g, t, ϕ, n_fermions)
    vs  = collect(vertices(g))
    pos = Dict(v => i for (i, v) in enumerate(vs))
    N   = length(vs)
    t1, λ1 = real(exp(im * ϕ)), imag(exp(im * ϕ))   # same parametrization as your gates
    h = zeros(ComplexF64, N, N)
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        a, b   = pos[v1], pos[v2]
        tij    = t * kagome_amp(v1, v2, t1, λ1)       # coeff of c†_{v1} c_{v2}
        h[a, b] += tij
        h[b, a] += conj(tij)                          # h.c.  -> h Hermitian
    end
    ε    = eigvals(Hermitian(h))                      # real, ascending
    E_gs = sum(@view ε[1:n_fermions])
    return E_gs, ε
end

function main_fermions(χ)

    #g = named_hexagonal_lattice_graph(4,4; periodic = false)
    g = named_kagome_lattice_graph(4,4)
    s = siteinds("fermion", g)
    ψ = fermionic_tensornetworkstate(ComplexF64,  v-> last(v) != 1 ? "Emp" : "Occ", g, s)
    n_fermions = length(filter(v -> last(v)== 1, collect(vertices(g))))
    println("Fermion density is $(n_fermions / length(vertices(g)))")
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    println("Imaginary time Evo to find spinless fermion GS of $(length(vertices(g))) sites with BP")
    dt = 0.01

    ϕ = pi/5
    t = 1
    ec = edge_color(g, 4)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14, normalize_tensors = true)
    #single_site_gates = [("RInt", v, 0.5*U*dt) for v in vertices(g)]
    #single_site_gates = [single_site_gates; [("RN", v, -0.25*U*dt) for v in vertices(g)]]
    two_site_gates =kagome_gates(s, g, dt, ϕ, t)

    nsteps = 1000

    e_gs, _ = free_fermion_gs_energy(g, t, ϕ, n_fermions)

    e_bp = bp_energy(ψ_bpc, t, ϕ)
    println("Initial BP energy density is $(e_bp / length(vertices(g)))")
    imaginary_times = ComplexF64[]
    energies = ComplexF64[]
    for i in 1:nsteps
        t1 = time()
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)

        if i % 1 == 0
            ψ_bpc = update(ψ_bpc)
            e_bp = bp_energy(ψ_bpc, t, ϕ)
            push!(energies, e_bp)
            push!(imaginary_times, i * abs(dt))
            println("Imaginary time is $(i * abs(dt))")
            println("BP energy density is $(e_bp / length(vertices(g)))")
            println("Actual GS energy density is $(e_gs / length(vertices(g)))")

            n_tot = sum([expect(ψ_bpc, (["N"], [v])) for v in vertices(g)])

            println("Total N density is $(n_tot / length(vertices(g)))")
        end
    end
end

χ = 4
main_fermions(χ)


