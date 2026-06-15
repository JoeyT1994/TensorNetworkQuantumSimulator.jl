using ITensors: dim

# =============================================================================
# Infinite tensor-network state: 2-site unit cell (sublattices :A and :B)
# connected by `z` directional bonds.
#
# The `z` bonds are encoded as `z` little "bond vertices" each carrying an
# identity, so that an environment approximation (e.g. belief propagation)
# produces one *independent* environment per bond (collapsing to a literal
# 2-vertex graph would give a single joint environment over all bonds, which
# couples them incorrectly). The wrapper below hides those bond vertices: you
# only ever name sites (:A / :B) and bonds (1:z).
# =============================================================================

# --- The unit-cell graph -----------------------------------------------------

# Vertex naming: site "A", site "B", bond k "AB$k" (e.g. "AB1", "AB2", ...).
const _A = "A"
const _B = "B"
_bondvert(k::Int) = "AB$k"

function unit_cell(z::Int)
    g = NamedGraph()
    add_vertex!(g, _A)
    add_vertex!(g, _B)
    for k in 1:z
        add_vertex!(g, _bondvert(k))
        add_edge!(g, _A, _bondvert(k))
        add_edge!(g, _B, _bondvert(k))
    end
    return g
end

# --- The clean wrapper -------------------------------------------------------

"""
    InfiniteTensorNetworkState

A translation-invariant infinite tensor-network *state* with a two-site unit cell
(sublattices `:A` and `:B`) connected by `z` directional bonds. It is just the
state (it wraps a `TensorNetworkState` on the unit-cell graph): address its sites
with `:A`/`:B` and its bonds with `1:z`, apply gates with `iTNS_apply_gate`, and
measure with `iTNS_expect` / `iTNS_reduced_density_matrix`.
"""
struct InfiniteTensorNetworkState{V}
    tns::TensorNetworkState{V}
end
const InfiniteTNS = InfiniteTensorNetworkState

tensornetworkstate(itns::InfiniteTensorNetworkState) = itns.tns
graph(itns::InfiniteTensorNetworkState) = graph(itns.tns)
siteinds(itns::InfiniteTensorNetworkState) = siteinds(itns.tns)
siteinds(itns::InfiniteTensorNetworkState, v) = siteinds(itns.tns, v)
Base.getindex(itns::InfiniteTensorNetworkState, v) = itns.tns[v]
Base.copy(itns::InfiniteTensorNetworkState) = InfiniteTensorNetworkState(copy(itns.tns))
coordination(itns::InfiniteTensorNetworkState) = degree(graph(itns.tns), _A)

# wrap the state in a BP cache (the cache holds the genuine TensorNetworkState, so
# all stock BP / measurement machinery works on it unchanged)
BeliefPropagationCache(itns::InfiniteTensorNetworkState) = BeliefPropagationCache(itns.tns)

# vertices for a location: a site, or both sites + the bond between them.
# These follow the fixed unit-cell naming convention, so they work on any cache
# built from an InfiniteTensorNetworkState.
_site(s::Symbol) = s === :A ? _A : s === :B ? _B : error("site must be :A or :B, got $s")
_locverts(s::Symbol) = [_site(s)]
_locverts(k::Int) = [_A, _bondvert(k), _B]
_obsverts(s::Symbol) = [_site(s)]
_obsverts(::Int) = [_A, _B]

"""
    infinite_tensornetworkstate(z; eltype=ComplexF64, sitetype="S=1/2", siteinds=nothing, init=v->"Up")

Build a product-state `InfiniteTensorNetworkState` with two sites and `z` bonds.
`init` maps a site (`:A`/`:B`) to a local state name/vector understood by
`ITensors.state`. By default the two sites get a single `sitetype` index each;
pass `siteinds` (a dictionary with `siteinds["A"]` and `siteinds["B"]` each a
`Vector{<:Index}`) to supply the physical indices yourself.
"""
function infinite_tensornetworkstate(
        z::Int;
        eltype = ComplexF64,
        sitetype::String = "S=1/2",
        siteinds = nothing,
        init = v -> "Up",
    )
    g = unit_cell(z)

    # physical indices on the two sites, none on the bond vertices
    sinds = Dictionary{vertextype(g), Vector{<:Index}}()
    if siteinds === nothing
        d, tag = site_dimension(sitetype), site_tag(sitetype)
        set!(sinds, _A, [Index(d, tag)])
        set!(sinds, _B, [Index(d, tag)])
    else
        set!(sinds, _A, collect(Index, siteinds[_A]))
        set!(sinds, _B, collect(Index, siteinds[_B]))
    end
    for k in 1:z
        set!(sinds, _bondvert(k), Index[])
    end

    # distinct bond-1 indices on each side of every bond, bridged by an identity
    linkA = [Index(1, "Link,b$k") for k in 1:z]
    linkB = [Index(1, "Link,b$k") for k in 1:z]

    tensors = Dictionary{vertextype(g), ITensor}()
    for (v, links) in ((_A, linkA), (_B, linkB))
        t = adapt(eltype)(ITensors.state(init(v), only(sinds[v])))
        for l in links
            t *= ITensors.onehot(eltype, l => 1)
        end
        set!(tensors, v, t)
    end
    for k in 1:z
        set!(tensors, _bondvert(k), adapt(eltype)(ITensors.denseblocks(ITensors.delta(linkA[k], linkB[k]))))
    end

    ψ = TensorNetworkState(TensorNetwork(tensors, g), sinds)
    return InfiniteTensorNetworkState(ψ)
end

const infinite_tns = infinite_tensornetworkstate

# --- the FERMIONIC unit-cell state -------------------------------------------

# Resolve a local fermionic state spec (a basis-state name like "0"/"Occ"/"Up", or a
# parity-definite vector) to its component vector on site index `s`.
_fermionic_local_vec(x::AbstractString, s::Index) = fermionic_statevector(String(x), s)
_fermionic_local_vec(x::AbstractVector{<:Number}, ::Index) = collect(x)
_fermionic_local_vec(x, ::Index) = error(
    "Unrecognized local fermion state $(repr(x)). Use a basis-state name (e.g. \"0\", \"Occ\", \"Up\") or a parity-definite vector."
)

"""
    infinite_fermionic_tensornetworkstate(z; eltype=ComplexF64, sitetype="fermion", init=v->"0")

Build a product-state fermionic `InfiniteTensorNetworkState` with two sites (`:A`/`:B`) and
`z` bonds, the locally-ordered (arXiv:2410.02215) analogue of [`infinite_tensornetworkstate`].
The tensors are `FermionicITensor`s; `init` maps a site (`:A`/`:B`) to a local basis-state name
(spinless: `"0"`/`"Occ"`; spinful: `"0"`/`"Up"`/`"Dn"`/`"UpDn"`) or a parity-definite vector.
`sitetype` is `"fermion"` (spinless, dim 2) or `"spinful_fermion"` (dim 4).

Because every one of the `z` bonds couples the two sublattices symmetrically (each bond vertex
holds a parity-even identity tieing site `:A` to site `:B`), a product state on this unit cell
requires sites `:A` and `:B` to carry **equal** fermion parity; the occupation parity is then
threaded onto a single odd dimension-1 bond (a Jordan–Wigner string), exactly as in
[`fermionic_tensornetworkstate`]. A mismatched parity (e.g. `:A` occupied, `:B` empty) cannot be
represented by parity-even unit-cell tensors and raises an error.
"""
function infinite_fermionic_tensornetworkstate(
        z::Int;
        eltype = ComplexF64,
        sitetype::String = "fermion",
        init = v -> "0",
    )
    g = unit_cell(z)
    d, tag = site_dimension(sitetype), site_tag(sitetype)

    # physical indices on the two sites, none on the bond vertices
    sinds = Dictionary{vertextype(g), Vector{<:Index}}()
    set!(sinds, _A, [Index(d, tag)])
    set!(sinds, _B, [Index(d, tag)])
    for k in 1:z
        set!(sinds, _bondvert(k), Index[])
    end
    sA, sB = only(sinds[_A]), only(sinds[_B])
    grA, grB = _fermionic_site_grading(sA), _fermionic_site_grading(sB)

    # local state vectors and their (definite) parities
    vecA = _fermionic_local_vec(init(:A), sA)
    vecB = _fermionic_local_vec(init(:B), sB)
    length(vecA) == d || error("Local state on :A has length $(length(vecA)) ≠ site dimension $d.")
    length(vecB) == d || error("Local state on :B has length $(length(vecB)) ≠ site dimension $d.")
    pA = fermionic_state_parity(vecA, grA)
    pB = fermionic_state_parity(vecB, grB)
    pA == pB || error(
        "The two-site / z-bond unit cell requires sites :A and :B to have EQUAL fermion parity " *
        "(each bond couples the sublattices symmetrically). Got parity(:A) = $(pA), parity(:B) = $(pB). " *
        "Pick local states of matching parity (e.g. both occupied or both empty)."
    )

    # Each bond carries dim-1 indices on either side; their parity must XOR to the site parity
    # so every vertex tensor is parity even. Put all the parity on bond 1 (a JW string), rest even.
    bond_parity = Bool[k == 1 ? pA : false for k in 1:z]
    linkA = [Index(1, "Fermion,Link,b$k") for k in 1:z]
    linkB = [Index(1, "Fermion,Link,b$k") for k in 1:z]

    grading = Dictionary{Index, Vector{Bool}}()
    set!(grading, sA, grA)
    set!(grading, sB, grB)
    for k in 1:z
        set!(grading, linkA[k], Bool[bond_parity[k]])
        set!(grading, linkB[k], Bool[bond_parity[k]])
    end

    # site tensors: state vector on the site leg, a 1 on each (dim-1) incident bond. Site and
    # bond legs all point OUT at the site (arrow convention: site → bond vertex).
    tensors = Dictionary{vertextype(g), FermionicITensor}()
    for (v, sind, vec, links) in ((_A, sA, vecA, linkA), (_B, sB, vecB, linkB))
        t = adapt(eltype)(ITensor(vec, sind))
        for l in links
            t *= ITensors.onehot(eltype, l => 1)
        end
        order = Index[sind; links]
        dirs = fill(false, length(order))
        gr = Dictionary{Index, Vector{Bool}}(order, Vector{Bool}[grading[i] for i in order])
        set!(tensors, v, FermionicITensor(t, order, dirs, gr))
    end

    # bond-vertex tensors: parity-even identity tieing linkA[k] (in) to linkB[k] (in), so the
    # arrows point :A → ABk and :B → ABk (matching the site tensors' out legs).
    for k in 1:z
        t = adapt(eltype)(ITensors.denseblocks(ITensors.delta(linkA[k], linkB[k])))
        order = Index[linkA[k], linkB[k]]
        gr = Dictionary{Index, Vector{Bool}}(order, Vector{Bool}[grading[linkA[k]], grading[linkB[k]]])
        set!(tensors, _bondvert(k), FermionicITensor(t, order, Bool[true, true], gr))
    end

    ψ = TensorNetworkState(TensorNetwork(tensors, g), sinds)
    return InfiniteTensorNetworkState(ψ)
end

const infinite_fermionic_tns = infinite_fermionic_tensornetworkstate
