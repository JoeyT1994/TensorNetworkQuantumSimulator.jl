# =============================================================================
# Embedding an InfiniteTensorNetworkState into a finite, periodic lattice patch.
#
# The iTNS carries a two-site (`:A`/`:B`) unit cell joined by `z` *directional*
# bonds: bond `k` is a specific lattice direction, and we do NOT assume any
# point-group symmetry, so bonds in different directions may carry different
# tensors / bond dimensions. To measure a quantity that needs genuine lattice
# loops (e.g. the loop-corrected free-energy density) we tile that unit cell onto
# a finite periodic patch of a *named* lattice of matching coordination: the A
# tensor on every A-sublattice site, the B tensor on every B-sublattice site, and
# a copy of bond-vertex `k` on every lattice edge whose direction is `k`.
#
# Index / direction bookkeeping (the part that must be exactly right):
#   * bond index `k` maps to lattice displacement `d_k` for an A site and to
#     `-d_k` for a B site — the same physical bond seen from its two ends. So an
#     A site at `p` connects through bond `k` to the B site at `p + d_k`, and that
#     B site connects back through its OWN bond `k` to `p`. Both ends agree on
#     `k`, which is exactly what lets bond-vertex `k`'s tensor sit on the edge
#     unchanged (no C4v / rotational symmetry assumed — only this directional
#     consistency).
#   * every site and bond tensor is copied with FRESH indices (`sim`), wired so
#     that an A-site's bond-`k` leg equals the bond vertex's A leg, and the bond
#     vertex's B leg equals the B-site's bond-`k` leg. Distinct sites therefore
#     never accidentally share an index, while each genuine bond is one shared
#     index (per network layer).
#
# Keeping the explicit bond vertices (rather than collapsing each bond to a single
# shared index) makes the tiled patch structurally identical to the unit cell, so
# ALL the stock machinery — belief propagation, the fermionic arrows / supertrace
# bookkeeping, `expect`, `norm_sqr`, the loop expansion — runs on it unchanged.
# =============================================================================

"""
    NamedLattice

Enumeration of the lattices into which an [`InfiniteTensorNetworkState`] unit cell
can be embedded. Each lattice fixes a coordination number and a table mapping the
unit-cell bond index `k` to a lattice direction (`_lattice_directions`).
Implemented: `SquareLattice` (coordination 4) and `HexagonalLattice` (coordination 3,
embedded as the brick-wall honeycomb — see `_lattice_directions`).
"""
@enum NamedLattice SquareLattice HexagonalLattice

# Bond-index → A-site displacement table. Bond `k` of the unit cell IS, by
# convention, lattice direction `_lattice_directions(lat)[k]`. This is the single
# place that fixes "which iTNS bond is which lattice direction". A B site uses the
# opposite displacement `-d_k` for its own bond `k`.
function _lattice_directions(lat::NamedLattice)
    lat == SquareLattice && return [(1, 0), (0, 1), (-1, 0), (0, -1)]
    # Honeycomb as a brick-wall on the SAME parity-coloured integer grid: an A site bonds to
    # +x, -x and +y (all parity-flipping, so each lands on a B site); a B site uses the
    # negatives (-x, +x, -y). This is coordination 3 with hexagonal faces, e.g. the 6-cycle
    # (0,0)→(1,0)→(2,0)→(2,1)→(1,1)→(0,1)→(0,0). Crucially it tiles the same rectangular even
    # torus under the same (x+y)-parity A/B colouring as the square lattice, so
    # `lattice_patch`/`embed` need no special-casing — only this directions table differs.
    lat == HexagonalLattice && return [(1, 0), (-1, 0), (0, 1)]
    return error("Unsupported lattice $lat")
end

"""
    lattice_coordination(lat::NamedLattice)

Number of bonds per site of the named lattice (`= z`), which an embedded iTNS must
match (`coordination(itns) == lattice_coordination(lat)`).
"""
lattice_coordination(lat::NamedLattice) = length(_lattice_directions(lat))

# Vertex names for the patch (Strings, like the unit cell). Physical sites are
# named by their integer coordinate; bond vertices by the (canonically ordered)
# pair of sites they join plus the bond index.
_sitename(c::NTuple{2, Int}) = "$(c[1]),$(c[2])"
function _bondname(ca::NTuple{2, Int}, cb::NTuple{2, Int}, k::Int)
    lo, hi = _sitename(ca) <= _sitename(cb) ? (ca, cb) : (cb, ca)
    return "b$(k):$(_sitename(lo))|$(_sitename(hi))"
end

"""
    lattice_patch(lat::NamedLattice, dims::NTuple{2, Int})

Build a finite periodic patch of `lat` with `dims = (Lx, Ly)` sites per direction
and its A/B two-colouring. Returns a `NamedTuple`:

  * `graph`       — `NamedGraph{String}` of physical sites AND one bond vertex per
                    lattice edge (mirroring the iTNS unit-cell graph),
  * `sites`       — `Vector{String}` of physical-site vertex names,
  * `sublattice`  — `Dict{String, Symbol}` mapping a site to `:A` or `:B`,
  * `bonds`       — `Vector{@NamedTuple{vert::String, A::String, B::String, k::Int}}`,
                    one per lattice edge: the bond-vertex name, its A-/B-endpoint
                    site names, and the unit-cell bond index `k`.

`dims` must be even (so the `(x + y)`-parity two-colouring is consistent across the
periodic boundary) and at least 4 in each direction (smaller tori make a site its
own neighbour through the wrap).
"""
function lattice_patch(lat::NamedLattice, dims::NTuple{2, Int})
    Lx, Ly = dims
    (iseven(Lx) && iseven(Ly)) || error("Patch dims must be even for a consistent A/B colouring, got $dims.")
    (Lx >= 4 && Ly >= 4) || error("Patch dims must be ≥ 4 in each direction, got $dims.")
    dirs = _lattice_directions(lat)
    z = length(dirs)
    wrap(c) = (mod(c[1], Lx), mod(c[2], Ly))
    sublat(c) = iseven(c[1] + c[2]) ? :A : :B

    coords = [(x, y) for x in 0:(Lx - 1) for y in 0:(Ly - 1)]
    sites = [_sitename(c) for c in coords]
    sublattice = Dict(_sitename(c) => sublat(c) for c in coords)

    g = NamedGraph()
    for s in sites
        add_vertex!(g, s)
    end

    bonds = @NamedTuple{vert::String, A::String, B::String, k::Int}[]
    # Iterate A sites × bond directions: every lattice edge has exactly one A
    # endpoint, so each edge is generated exactly once with a well-defined `k`.
    for c in coords
        sublat(c) == :A || continue
        for k in 1:z
            cb = wrap((c[1] + dirs[k][1], c[2] + dirs[k][2]))
            an, bn = _sitename(c), _sitename(cb)
            w = _bondname(c, cb, k)
            add_vertex!(g, w)
            add_edge!(g, an, w)
            add_edge!(g, bn, w)
            push!(bonds, (vert = w, A = an, B = bn, k = k))
        end
    end
    return (graph = g, sites = sites, sublattice = sublattice, bonds = bonds)
end

# The unit-cell pieces an embedding needs: the A/B site tensors, the bond-vertex
# tensors, the two site indices, and the per-bond link indices on each side.
function _unit_cell_pieces(itns::InfiniteTensorNetworkState)
    z = coordination(itns)
    tA, tB = itns[_A], itns[_B]
    sA, sB = only(siteinds(itns, _A)), only(siteinds(itns, _B))
    tAB = [itns[_bondvert(k)] for k in 1:z]
    linkA = [commonind(tA, tAB[k]) for k in 1:z]
    linkB = [commonind(tB, tAB[k]) for k in 1:z]
    return (; z, tA, tB, sA, sB, tAB, linkA, linkB)
end

"""
    embed(itns::InfiniteTensorNetworkState, lat::NamedLattice, dims::NTuple{2, Int})

Tile the iTNS two-site unit cell onto the periodic `lat` patch of size `dims`,
returning an ordinary `TensorNetworkState` (bosonic `ITensor`s or fermionic
`FermionicITensor`s, matching `itns`). The A tensor is placed on every A site, the
B tensor on every B site, and bond-vertex `k`'s tensor on every direction-`k`
edge, all with fresh, consistently wired indices (see the file header).

The result is a genuine, translation-invariant finite state: belief propagation,
`expect`, `norm_sqr` and the loop expansion all apply to it directly.
"""
function embed(itns::InfiniteTensorNetworkState, lat::NamedLattice, dims::NTuple{2, Int})
    coordination(itns) == lattice_coordination(lat) ||
        error("iTNS coordination $(coordination(itns)) ≠ $lat coordination $(lattice_coordination(lat)).")
    p = lattice_patch(lat, dims)
    uc = _unit_cell_pieces(itns)

    T = typeof(itns[_A])                                   # ITensor or FermionicITensor
    V = vertextype(p.graph)                                # match the patch graph's vertex type
    tensors = Dictionary{V, T}()
    sinds = Dictionary{V, Vector{<:Index}}()

    # Fresh per-site link index for each (site, bond k); shared with the incident
    # bond vertex's matching leg below.
    linkindex = Dict{Tuple{V, Int}, Index}()

    for s in p.sites
        if p.sublattice[s] === :A
            sfresh = sim(uc.sA)
            newlinks = Index[sim(uc.linkA[k]) for k in 1:uc.z]
            t = replaceinds(uc.tA, Index[uc.sA; uc.linkA], Index[sfresh; newlinks])
        else
            sfresh = sim(uc.sB)
            newlinks = Index[sim(uc.linkB[k]) for k in 1:uc.z]
            t = replaceinds(uc.tB, Index[uc.sB; uc.linkB], Index[sfresh; newlinks])
        end
        for k in 1:uc.z
            linkindex[(s, k)] = newlinks[k]
        end
        set!(tensors, s, t)
        set!(sinds, s, Index[sfresh])
    end

    for b in p.bonds
        iA, iB = linkindex[(b.A, b.k)], linkindex[(b.B, b.k)]
        tw = replaceinds(uc.tAB[b.k], Index[uc.linkA[b.k], uc.linkB[b.k]], Index[iA, iB])
        set!(tensors, b.vert, tw)
        set!(sinds, b.vert, Index[])
    end

    return TensorNetworkState(TensorNetwork(tensors, p.graph), sinds)
end
