using ITensors: ITensors, ITensor, Index, dim
using NamedGraphs: NamedEdge

# A fermionic tensor network state is simply an ordinary `TensorNetworkState` whose tensors
# are `FermionicITensor`s. The `is_fermionic` trait (src/TensorNetworks/tensornetworkstate.jl)
# detects this and routes `norm_factors`/`norm_sqr`/`expect` to the fermionic bodies in
# observables.jl, so no dedicated state type is needed. Each `FermionicITensor` carries its
# own Z2-parity grading (per-component parity bits of each leg) and leg arrows; tensors are
# parity even. Site legs point out (this is a ket).

# Whole-network grading: merge every per-tensor grading. The bond Index objects are
# shared between the two endpoints' tensors, so `merge` collapses the duplicate keys.

function grading(ψ::TensorNetworkState, v)
    ψ[v] isa FermionicITensor && return ψ[v].grading
    error("Tensor is not fermionic and so doesn't have grading")
end

function grading(ψ::TensorNetworkState, e::NamedEdge)
    return grading(ψ, src(e))[virtualind(ψ, e)]
end

grading(ψ::TensorNetworkState) = reduce(merge, [grading(ψ, v) for v in vertices(ψ)])

# Convenience: the per-vertex FermionicITensor is just the underlying tensor.
function FermionicITensor(ψ::TensorNetworkState, v)
    ψ[v] isa FermionicITensor && return ψ[v]
    error("Tensor is not fermionic")
end

# Z2 grading (per-component parity bits) of a fermionic site index.
function _fermionic_site_grading(sind::Index)
    if dim(sind) == 2 && occursin("fermion", string(tags(sind)))
        return [false, true]                       # |0⟩ even, |1⟩ odd
    elseif dim(sind) == 4 && occursin("spinful_fermion", string(tags(sind)))
        return [false, true, true, false]          # |0⟩, |↑⟩, |↓⟩, |↑↓⟩
    end
    error("Don't recognize this as a fermionic site index")
end

# Build the locally-ordered leg metadata shared by every fermionic `TensorNetworkState`:
#   * `grading`          : parity bits for every site and bond `Index`,
#   * `index_order`      : per-vertex fermionic leg order `[site(s); incident bonds]`,
#   * `index_directions` : per-vertex arrows (sites out/`false`; each bond out at `src`,
#                          in at `dst`, so the arrow points `src → dst`),
#   * `l`                : `NamedEdge → bond Index` (stored for both orientations).
# `bond_grading(e)` returns the parity-bit vector of edge `e`'s bond; its length is the
# bond dimension. Arrows are local to bonds, so no global tensor ordering is needed
# (locally-ordered formalism, arXiv:2410.02215).
function _fermionic_scaffolding(g::AbstractGraph, siteinds::Dictionary, bond_grading::Function)
    grading = Dictionary{Index, Vector{Bool}}()
    index_order = Dictionary{vertextype(g), Vector{<:Index}}()
    index_directions = Dictionary{vertextype(g), Vector{Bool}}()

    for v in collect(vertices(g))
        sinds = siteinds[v]
        for sind in sinds
            set!(grading, sind, _fermionic_site_grading(sind))
        end
        set!(index_order, v, collect(sinds))
        set!(index_directions, v, fill(false, length(sinds)))
    end

    l = Dict{NamedEdge{vertextype(g)}, Index}()
    for e in edges(g)
        bg = bond_grading(e)
        b = Index(length(bg), "Fermion,Link")
        set!(grading, b, bg)
        index_order[src(e)] = [index_order[src(e)]; [b]]
        index_order[dst(e)] = [index_order[dst(e)]; [b]]
        index_directions[src(e)] = [index_directions[src(e)]; [false]]
        index_directions[dst(e)] = [index_directions[dst(e)]; [true]]
        l[e] = b
        l[reverse(e)] = b
    end

    return grading, index_order, index_directions, l
end

"""
    random_fermionic_tensornetworkstate(eltype, g::AbstractGraph; bond_dimension = 1, bond_grading = ...)

Generate a random parity-even fermionic `TensorNetworkState` on graph `g`.

Each vertex carries a single spinless-fermion site index (dimension 2, grading `[false, true]`
i.e. `|0⟩` even / `|1⟩` odd). Neighbouring tensors share one graded virtual (bond) index of
dimension `bond_dimension`, whose even/odd split is given by `bond_grading`. Every tensor is
made parity even by zeroing odd-total-parity components.

This is the locally-ordered formalism (arXiv:2410.02215): each tensor stores its own leg
`index_order`, and each shared bond carries an arrow via `index_directions` (`true` = in/−,
`false` = out/+). For every edge the `src` endpoint holds the bond as out and the `dst`
endpoint as in, so the arrow points `src → dst`. Site legs point out. Fermionic signs are
resolved per-bond at contraction time by [`contract`](@ref) — there is no global
ordering of the tensors.

# Keyword Arguments
- `bond_dimension::Integer`: Virtual bond dimension (default `1`).
- `bond_grading::Vector{Bool}`: Per-component Z2 parity of the bond index (default splits the
  dimension into `⌈d/2⌉` even then `⌊d/2⌋` odd components).
"""
function random_fermionic_tensornetworkstate(
        eltype, g::AbstractGraph,
        siteinds::Dictionary = siteinds("fermion", g);
        bond_dimension::Integer = 1,
        bond_grading::Vector{Bool} = Vector{Bool}([falses(cld(bond_dimension, 2)); trues(fld(bond_dimension, 2))]),
    )
    vs = collect(vertices(g))

    # Every bond shares the same dimension and grading.
    grading, index_order, index_directions, l = _fermionic_scaffolding(g, siteinds, e -> bond_grading)

    tensors = Dictionary{vertextype(g), FermionicITensor}()
    for v in vs
        is = Index[siteinds[v]; [l[NamedEdge(v => vn)] for vn in neighbors(g, v)]]
        t = random_even_itensor(eltype, is, grading)
        local_grading = Dictionary{Index, Vector{Bool}}(is, [grading[i] for i in is])
        ft = FermionicITensor(t, index_order[v], index_directions[v], local_grading)
        set!(tensors, v, ft)
    end


    return TensorNetworkState(TensorNetwork(tensors, g), siteinds)
end

function random_fermionic_tensornetworkstate(g::AbstractGraph, args...; kwargs...)
    return random_fermionic_tensornetworkstate(Float64, g, args...; kwargs...)
end

"""
    fermionic_tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary = siteinds("fermion", g))

Construct a fermionic product `TensorNetworkState` on graph `g`, where `f` maps each vertex
to its local state. Local states may be given as basis-state Strings (spinless: `"0"`/`"Emp"`,
`"1"`/`"Occ"`; spinful: `"0"`/`"Emp"`, `"Up"`/`"↑"`, `"Dn"`/`"↓"`, `"UpDn"`/`"↑↓"`) or as
parity-definite vectors.

Each tensor is parity even (the locally-ordered formalism, arXiv:2410.02215). Because an
occupied site (`|1⟩`, `|↑⟩`, `|↓⟩`) is parity *odd*, the occupation parity is carried on
dimension-1 *odd* virtual bonds: a Jordan–Wigner-style string wired along a spanning tree.
The required odd-bond set is the unique T-join on the tree, computed in one post-order pass
(a bond is odd iff the total occupation parity of the subtree below it is odd). Loop (non-tree)
edges carry even dimension-1 bonds.

This requires an **even** total number of fermions: an odd total parity cannot be represented
by parity-even tensors alone and raises an error.
"""
function fermionic_tensornetworkstate(
        eltype, f::Function, g::AbstractGraph,
        siteinds::Dictionary = siteinds("fermion", g),
    )
    vs = collect(vertices(g))

    # 1) Resolve each vertex's local state vector and its (definite) parity.
    state_vecs = Dictionary{vertextype(g), Vector}()
    p = Dictionary{vertextype(g), Bool}()
    for v in vs
        sind = only(siteinds[v])
        gr = _fermionic_site_grading(sind)
        fv = f(v)
        vec = if fv isa String
            fermionic_statevector(fv, sind)
        elseif fv isa AbstractVector{<:Number}
            collect(fv)
        else
            error("Unrecognized local state constructor. Supported: String names and parity-definite vectors.")
        end
        length(vec) == dim(sind) || error("Local state vector length $(length(vec)) ≠ site dimension $(dim(sind)).")
        set!(state_vecs, v, vec)
        set!(p, v, fermionic_state_parity(vec, gr))
    end

    # 2) Hard error on odd total fermion parity — not representable by parity-even tensors.
    isodd(sum(values(p))) && error("Total fermion parity is odd: a product state with an odd number of fermions cannot be built from parity-even tensors in the locally-ordered formalism.")

    # 3) T-join on a spanning tree: a dim-1 bond is ODD iff the occupation parity of the
    #    subtree below it is odd. `post_order_dfs_edges` returns tree edges directed
    #    child → parent (loop edges excluded); one pass folds subtree parities upward.
    acc = Dictionary(vs, [p[v] for v in vs])
    odd_bond = Dict{NamedEdge{vertextype(g)}, Bool}()
    for e in post_order_dfs_edges(g, first(vs))
        c, par = src(e), dst(e)
        odd_bond[e] = acc[c]
        odd_bond[reverse(e)] = acc[c]
        acc[par] ⊻= acc[c]
    end

    # 4) Scaffolding with per-edge bond grading from the T-join (loop edges default even).
    bond_grading(e) = get(odd_bond, e, false) ? Bool[true] : Bool[false]
    grading, index_order, index_directions, l = _fermionic_scaffolding(g, siteinds, bond_grading)

    # 5) Build each parity-even vertex tensor: state vector on the site leg, a 1 on every
    #    dimension-1 incident bond.
    tensors = Dictionary{vertextype(g), FermionicITensor}()
    for v in vs
        sind = only(siteinds[v])
        t = adapt(eltype)(ITensor(state_vecs[v], sind))
        for vn in neighbors(g, v)
            t *= onehot(eltype, l[NamedEdge(v => vn)] => 1)
        end
        is = collect(index_order[v])
        local_grading = Dictionary{Index, Vector{Bool}}(is, [grading[i] for i in is])
        ft = FermionicITensor(t, index_order[v], index_directions[v], local_grading)
        set!(tensors, v, adapt(eltype)(ft))
    end

    return TensorNetworkState(TensorNetwork(tensors, g), siteinds)
end

function fermionic_tensornetworkstate(f::Function, args...)
    return fermionic_tensornetworkstate(Float64, f, args...)
end

# Fermionic analogue of the bosonic `norm_factors` (src/TensorNetworks/tensornetworkstate.jl):
# for each vertex emit its ket tensor, its bra tensor, and any single-site operator factor,
# producing the flat `2N + n_ops` parity-even tensor list for ⟨ψ|O|ψ⟩ (unnormalised). The
# caller folds the list with the order-independent `contract`; the optimal sequence finder
# picks the fold tree. `op_strings(v)` gives the operator name on vertex `v` ("I"/"ρ" =
# identity). Returns `nothing` when `O` has odd total parity (⟨O⟩ = 0).
# Per-vertex grouped form of `fermionic_norm_factors`: returns a `Dictionary` mapping each
# vertex in `verts` to its own `[ket, bra, (op)]` factor list, or `nothing` when the operator
# product has odd total parity (⟨O⟩ = 0). The single operator-string dummy bond `d` (Eq.95-96)
# shared by a pair of odd factors is created ONCE here and threaded into both vertices, so the
# grouping is safe to consume vertex-by-vertex (e.g. by the boundary-MPS `path_contract`): the
# `d` leg simply stays open on the running tensor until the second odd factor is contracted.
function fermionic_norm_factors_grouped(ψ::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I")
    odd_vs = [v for v in verts if is_odd(op_strings(v))]
    isodd(length(odd_vs)) && return nothing             # parity-forbidden ⇒ ⟨O⟩ = 0
    length(odd_vs) > 2 && error("Fermionic expect currently supports at most one pair of odd operators (e.g. a single hopping term).")
    d = isempty(odd_vs) ? nothing : Index(1, "Fermion,OpString")

    grouped = Dictionary{eltype(verts), Vector{FermionicITensor}}()
    for v in verts
        name = op_strings(v)
        op_here = !(name == "I" || name == "ρ")
        ket = ψ[v]
        # keep the un-operated site legs unprimed; prime everything else (bonds, op sites)
        keep = op_here ? Index[] : collect(siteinds(ψ, v))
        fs = FermionicITensor[ket, dag(ket, keep)]
        if op_here
            s = only(siteinds(ψ, v))
            (dim(s) == 2 || dim(s) == 4) || error("Fermionic measurement currently supports spinless (dimension-2) or spinful (dimension-4) sites only.")
            sgr = ket.grading[s]
            # A pair of ODD factors (e.g. a hopping c_i† c_j) shares the dummy `d`; each carries
            # it with the arrow set by `odd_op_tensor` (creation = bra, annihilation = ket), so
            # `contract` inserts the supertrace g once — supplying the (−1) of the fermionic
            # operator ordering automatically, regardless of fold order.
            push!(fs, is_odd(name) ? odd_op_tensor(s, name, d, sgr) : even_op_tensor(s, name, sgr))
        end
        set!(grouped, v, fs)
    end
    return grouped
end

# Fermionic analogue of the bosonic `norm_factors` (src/TensorNetworks/tensornetworkstate.jl):
# for each vertex emit its ket tensor, its bra tensor, and any single-site operator factor,
# producing the flat `2N + n_ops` parity-even tensor list for ⟨ψ|O|ψ⟩ (unnormalised). The
# caller folds the list with the order-independent `contract`; the optimal sequence finder
# picks the fold tree. `op_strings(v)` gives the operator name on vertex `v` ("I"/"ρ" =
# identity). Returns `nothing` when `O` has odd total parity (⟨O⟩ = 0).
function fermionic_norm_factors(ψ::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I")
    grouped = fermionic_norm_factors_grouped(ψ, verts; op_strings)
    grouped === nothing && return nothing
    return reduce(vcat, [grouped[v] for v in verts]; init = FermionicITensor[])
end