# ===========================================================================
# Chiral Kagome ground state by TRIANGLE-EXACT cluster updates, in the
# Kagome -> honeycomb incidence-graph picture, with OPEN boundaries.
#
#   * leaf   = a Kagome site  (one spinless-fermion mode)
#   * centre = a Kagome triangle / honeycomb vertex (NO-MODE virtual hub)
#
# Sweep (imaginary time):  all UP triangles at once -> all DOWN clusters at
# once -> run BP -> measure BP energy.  "Rinse and repeat."
#
# Edge bookkeeping (so the cluster decomposition is the EXACT open Hamiltonian):
#   up layer   : every intra-cell triangle (always a complete 3-clique) -> 3-site gate
#   down layer : complete down-triangles (i,j>=2)                       -> 3-site gate
#                + boundary "dangling" edges (single edges at i=1/j=1)  -> 2-site hop gate
# Up/down clusters are each a vertex-disjoint matching, so a whole layer commutes
# and is applied "at once". Every Kagome edge lands in exactly one cluster, so
# `free_fermion_gs_energy(gk, ...)` on the full open graph is the exact ED reference.
# ===========================================================================

using TensorNetworkQuantumSimulator
const TNS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: fermionic_hopping_gate
using NamedGraphs: NamedGraph, NamedEdge, add_edge!
using Graphs: neighbors, has_edge
using ITensors: Index
using LinearAlgebra
using Serialization

include(joinpath(@__DIR__, "kagome_cluster_helpers.jl"))

# ---------------------------------------------------------------------------
# open-boundary Kagome lattice + chiral amplitude + ED reference
# (copied from examples/fermionic_benchmarks/kagome_gs.jl so this driver is standalone)
# ---------------------------------------------------------------------------
function named_kagome_lattice_graph(nx::Int, ny::Int)
    verts = [(i, j, s) for i in 1:nx for j in 1:ny for s in 1:3]
    g = NamedGraph(verts)
    inrange(i, j) = 1 <= i <= nx && 1 <= j <= ny
    for i in 1:nx, j in 1:ny
        g = add_edge!(g, NamedEdge((i, j, 1) => (i, j, 2)))
        g = add_edge!(g, NamedEdge((i, j, 2) => (i, j, 3)))
        g = add_edge!(g, NamedEdge((i, j, 3) => (i, j, 1)))
        inrange(i + 1, j)     && (g = add_edge!(g, NamedEdge((i, j, 2) => (i + 1, j, 1))))
        inrange(i, j + 1)     && (g = add_edge!(g, NamedEdge((i, j, 3) => (i, j + 1, 1))))
        inrange(i + 1, j - 1) && (g = add_edge!(g, NamedEdge((i, j, 2) => (i + 1, j - 1, 3))))
    end
    return g
end

# coeff of c†_{v1} c_{v2}: along-arrow (A→B→C→A) amplitude (t1 - im λ1), else (t1 + im λ1)
function kagome_amp(v1, v2, t1, λ1)
    s1, s2 = v1[3], v2[3]
    along = (s2, s1) in ((1, 2), (2, 3), (3, 1))
    return along ? (t1 - im * λ1) : (t1 + im * λ1)
end

function free_fermion_gs_energy(g, t, ϕ, n_fermions)
    vs  = collect(vertices(g))
    pos = Dict(v => i for (i, v) in enumerate(vs))
    N   = length(vs)
    t1, λ1 = real(exp(im * ϕ)), imag(exp(im * ϕ))
    h = zeros(ComplexF64, N, N)
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        a, b   = pos[v1], pos[v2]
        tij    = t * kagome_amp(v1, v2, t1, λ1)
        h[a, b] += tij
        h[b, a] += conj(tij)
    end
    ε    = eigvals(Hermitian(h))
    return sum(@view ε[1:n_fermions]), ε
end

# ---------------------------------------------------------------------------
# triangle / dangling-edge enumeration on the Kagome graph
# ---------------------------------------------------------------------------
leafv(kv) = (:s, kv[1], kv[2], kv[3])               # Kagome site -> incidence leaf vertex
canon(a, b) = a <= b ? (a, b) : (b, a)

# all 3-cliques of `g`, each as a sorted 3-tuple of Kagome vertices
function find_triangles(g)
    tris = Set{NTuple{3, NTuple{3, Int}}}()
    for v in vertices(g)
        nb = collect(neighbors(g, v))
        for i in 1:length(nb), j in (i + 1):length(nb)
            if has_edge(g, nb[i], nb[j])
                push!(tris, Tuple(sort([v, nb[i], nb[j]])))
            end
        end
    end
    return collect(tris)
end

is_up(tri) = all(t -> t[1] == tri[1][1] && t[2] == tri[1][2], tri)   # all three share cell (i,j)

# ---------------------------------------------------------------------------
# build the incidence graph + per-cluster data (centre vertex, leaf vertices,
# physical site indices, and the imaginary-time gate)
# ---------------------------------------------------------------------------
struct Cluster
    centre::NTuple{4, Any}
    leafverts::Vector
    sites::Vector{Index}
    gate::FermionicITensor
end

function build_kagome_cluster_problem(nx, ny, dt, ϕ, t_hop; μ = 0.0)
    gk = named_kagome_lattice_graph(nx, ny)
    t1, λ1 = real(exp(im * ϕ)), imag(exp(im * ϕ))

    tris = find_triangles(gk)
    up_tris   = [t for t in tris if is_up(t)]
    down_tris = [t for t in tris if !is_up(t)]

    tri_edge_set = Set{Tuple}()
    for t in tris
        push!(tri_edge_set, canon(t[1], t[2]))
        push!(tri_edge_set, canon(t[1], t[3]))
        push!(tri_edge_set, canon(t[2], t[3]))
    end
    dangling = [e for e in edges(gk) if !(canon(src(e), dst(e)) in tri_edge_set)]

    # ---- incidence graph vertices: all leaves + a centre per cluster ----
    leaves = [leafv(kv) for kv in vertices(gk)]
    up_centre(t)   = (:up, t[1][1], t[1][2], 0)
    down_centres   = [(:dn, k, 0, 0) for k in 1:length(down_tris)]
    dang_centres   = [(:dh, k, 0, 0) for k in 1:length(dangling)]

    all_verts = Any[leaves...]
    append!(all_verts, [up_centre(t) for t in up_tris])
    append!(all_verts, down_centres)
    append!(all_verts, dang_centres)

    gi = NamedGraph(all_verts)
    add_inc!(c, lv) = (gi = add_edge!(gi, NamedEdge(c => lv)))
    for t in up_tris;   c = up_centre(t);     for kv in t; add_inc!(c, leafv(kv)); end; end
    for (k, t) in enumerate(down_tris); c = down_centres[k]; for kv in t; add_inc!(c, leafv(kv)); end; end
    for (k, e) in enumerate(dangling);  c = dang_centres[k]; add_inc!(c, leafv(src(e))); add_inc!(c, leafv(dst(e))); end

    # ---- site indices: leaves carry a fermion mode, centres are no-mode hubs ----
    sinds = siteinds("fermion", gi)
    for c in [up_centre.(up_tris); down_centres; dang_centres]
        sinds[c] = Index[]
    end

    # ---- build clusters with their imaginary-time gates ----
    # `μ_cl` is the chemical potential folded into THIS triangle's gate. Every Kagome site
    # belongs to exactly one up-triangle, so giving the full -μN to the up-triangles (and 0 to
    # the down clusters) realises the grand-canonical generator H_hop - μN with no double count.
    function tri_cluster(t, centre; μ_cl = 0.0)
        sl = [only(sinds[leafv(kv)]) for kv in t]
        ws = (t_hop * kagome_amp(t[1], t[2], t1, λ1),
              t_hop * kagome_amp(t[2], t[3], t1, λ1),
              t_hop * kagome_amp(t[3], t[1], t1, λ1))
        g  = triangle_gate(dt, (sl[1], sl[2], sl[3]), ws; coeff = -1.0, μ = μ_cl)
        return Cluster(centre, [leafv(kv) for kv in t], sl, g)
    end
    function hop_cluster(e, centre)
        va, vb = src(e), dst(e)
        sa, sb = only(sinds[leafv(va)]), only(sinds[leafv(vb)])
        g = fermionic_hopping_gate(dt, sa, sb; t = t_hop * kagome_amp(va, vb, t1, λ1), coeff = -1)
        return Cluster(centre, [leafv(va), leafv(vb)], Index[sa, sb], g)
    end

    up_clusters   = [tri_cluster(t, up_centre(t); μ_cl = μ) for t in up_tris]
    down_clusters = Cluster[]
    for (k, t) in enumerate(down_tris); push!(down_clusters, tri_cluster(t, down_centres[k])); end
    for (k, e) in enumerate(dangling);  push!(down_clusters, hop_cluster(e, dang_centres[k])); end

    return gk, gi, sinds, up_clusters, down_clusters
end

# ---------------------------------------------------------------------------
# one cluster update, written back into the BP cache's network in place
# ---------------------------------------------------------------------------
function apply_cluster!(ψ_bpc, cl::Cluster; maxdim, cutoff, gauge = true)
    net = TNS.network(ψ_bpc)
    centre = net[cl.centre]
    leaves = [net[lv] for lv in cl.leafverts]
    envs = gauge ? TNS.incoming_messages(ψ_bpc, Any[cl.centre; cl.leafverts...]) : nothing
    newleaves, newcentre, Ss = cluster_update_triangle(centre, leaves, cl.sites, cl.gate; cutoff, maxdim, envs)
    TNS.setindex_preserve!(ψ_bpc, newcentre, cl.centre)
    for (lv, t) in zip(cl.leafverts, newleaves)
        TNS.setindex_preserve!(ψ_bpc, t, lv)
    end
    # Re-home each emerging centre<->leaf bond spectrum as the BP message on BOTH edge
    # orientations (mirrors `apply_gate!`). The re-split mints fresh bond indices whose
    # dimension changes under truncation; without this the cache keeps a stale message on the
    # OLD bond, and the next `update(ψ_bpc)` reads it on the wrong bond — corrupting BP (the
    # multi-leg `incoming_messages` / `pseudo_sqrt_inv_sqrt` failures).
    for (k, lv) in enumerate(cl.leafverts)
        e = NamedEdge(cl.centre => lv)
        b = commoninds(newleaves[k], newcentre)
        isempty(b) && continue
        m, m_rev = TNS._bond_spectrum_messages(Ss[k], only(b), e)
        TNS.setmessage!(ψ_bpc, e, m)
        TNS.setmessage!(ψ_bpc, reverse(e), m_rev)
    end
    return ψ_bpc
end

# energy of the chiral Kagome Hamiltonian, measured on the incidence leaves
function energy_incidence(ψ_bpc, gk, t_hop, ϕ)
    t1, λ1 = real(exp(im * ϕ)), imag(exp(im * ϕ))
    e = 0.0 + 0.0im
    for ed in edges(gk)
        v1, v2 = src(ed), dst(ed)
        tij = kagome_amp(v1, v2, t1, λ1)
        l1, l2 = leafv(v1), leafv(v2)
        e += tij * expect(ψ_bpc, (["Cdag", "C"], [l1, l2])) +
             conj(tij) * expect(ψ_bpc, (["Cdag", "C"], [l2, l1]))
    end
    return t_hop * e
end

# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
function run(; nx = 2, ny = 2, dt = 0.05, nsteps = 200, χ = 16, cutoff = 1e-12,
        ϕ = pi / 5, t_hop = 1.0, measure_every = 20, measure_exact = false, gauge = true,
        μ = nothing)

    n_ferm = count(kv -> kv[3] == 1, vertices(named_kagome_lattice_graph(nx, ny)))  # 1/3 filling

    # Chemical potential that pins the target filling: the single-particle gap midpoint
    # (μ in the gap ⇒ the grand-canonical ground state of H_hop - μN has exactly n_ferm
    # fermions). Tracks ~-0.88 t at ϕ=π/5; pass `μ` explicitly to override, `μ = 0` to disable.
    if μ === nothing
        _, ε = free_fermion_gs_energy(named_kagome_lattice_graph(nx, ny), t_hop, ϕ, n_ferm)
        εs = real.(sort(ε))
        μ = (εs[n_ferm] + εs[n_ferm + 1]) / 2
    end

    gk, gi, sinds, up_clusters, down_clusters =
        build_kagome_cluster_problem(nx, ny, dt, ϕ, t_hop; μ)

    n_sites = length(vertices(gk))
    println("Kagome $(nx)x$(ny): $n_sites sites, $(length(up_clusters)) up-clusters, ",
            "$(length(down_clusters)) down-clusters (incl. dangling).")
    println("Filling = $(n_ferm)/$(n_sites) = $(round(n_ferm / n_sites; digits = 4)); ",
            "chemical potential μ = $(round(μ; digits = 5))")

    # 1/3-filled product initial state (fermion on every sublattice-1 leaf)
    ψ = fermionic_tensornetworkstate(ComplexF64,
        v -> (first(v) === :s && v[4] == 1) ? "Occ" : "Emp", gi, sinds)
    ψ_bpc = update(BeliefPropagationCache(ψ)); rescale!(ψ_bpc)

    ns = expect(ψ_bpc, [("N", leafv(kv)) for kv in vertices(gk)])

    e_gs, _ = free_fermion_gs_energy(gk, t_hop, ϕ, n_ferm)
    e0 = energy_incidence(ψ_bpc, gk, t_hop, ϕ)
    #n_tot = sum([expect(ψ_bpc, ("N", v)) for v in vertices(gk)])
    println("\nED   GS energy density        = $(round(e_gs / n_sites; digits = 8))")
    println("init BP energy density        = $(round(real(e0) / n_sites; digits = 8))")
    println("Filling is $(sum(ns) / n_sites)")
    #println("Initial filling $(n_tot / length(vertices(gk)))")
    println("\n step   τ        E_BP/site        E_GS_exact/site")
    for i in 1:nsteps
        for cl in up_clusters;   apply_cluster!(ψ_bpc, cl; maxdim = χ, cutoff, gauge); end
        for cl in down_clusters; apply_cluster!(ψ_bpc, cl; maxdim = χ, cutoff, gauge); end
        ψ_bpc = update(ψ_bpc); rescale!(ψ_bpc)
        if i % measure_every == 0 || i == nsteps
            e_bp = energy_incidence(ψ_bpc, gk, t_hop, ϕ)
            println(rpad(i, 6), " ", rpad(round(i * dt; digits = 3), 8), " ",
                rpad(round(real(e_bp) / n_sites; digits = 8), 16), " ",
                rpad(round(real(e_gs) / n_sites; digits = 8), 16), " ")
            ns = expect(ψ_bpc, [("N", leafv(kv)) for kv in vertices(gk)])
            println("Filling is $(sum(ns) / n_sites)")
        end
    end

    serialize("/Users/jtindall/Files/Data/Fermions/KagomeSpinlessGS/nx$(nx)ny$(ny)maxdim$(χ).jld2", ψ_bpc)
    return nothing
end

# ---------------------------------------------------------------------------
# Demo run. `gauge = true` gauges each cluster's external bonds by its incoming BP
# messages before the truncating re-split (Vidal gauge → ~variational); `gauge = false`
# truncates in the bare cluster norm. `χ = nothing, cutoff = 0.0` is a lossless evolution.
# 1/3 filling requires n_fermions = nx*ny to be EVEN (2x2 → 4 ✓, 4x4 → 16 ✓, 3x3 → 9 ✗).
# ---------------------------------------------------------------------------
χ = 8
run(; nx = 4, ny = 4, dt = 0.01, nsteps = 1000, χ, cutoff = 1e-14, measure_every = 10, gauge = true)
