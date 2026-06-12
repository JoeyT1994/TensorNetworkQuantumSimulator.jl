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
