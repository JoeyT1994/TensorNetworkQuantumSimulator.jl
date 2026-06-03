using ITensors: ITensors, ITensor, Index, dim
using NamedGraphs: NamedEdge

"""
    FermionicTensorNetworkState{V} <: AbstractTensorNetwork{V}

A fermionic tensor network state on a graph with vertices of type `V`. Behaves like a
[`TensorNetworkState`](@ref) but additionally tracks the data needed to resolve fermionic
minus signs on a network of *dense* ITensors: a Z2-parity grading per index and a global
fermionic order (`gpos`).

The grading assigns to every index `i` a `Vector{Bool}` of length `dim(i)` recording the
Z2 parity (`false` = even, `true` = odd) of each basis component. The global fermionic
order assigns every index a real-valued position; fermionic swap / Koszul signs are always
computed relative to this order (never the incidental ITensor leg order). Each tensor is
parity even. For now, assume siteinds point out (this is a ket)

# Fields
- `tensornetwork::TensorNetwork{V}`: The underlying (dense) tensor network.
- `siteinds::Dictionary{V, Vector{<:Index}}`: Physical (site) indices at each vertex.
- `index_order::Dictionary{V, Vector{<:Index}}: ordering of the indices for each tensor`
- `index_directions::Dictionary{V, Vector{Bool}}: direction of each of the indices for each tenor (true is in, false is out)``
- `grading::Dictionary{Index, Vector{Bool}}`: Maps each index `id` to its per-component Z2 parity.
"""
struct FermionicTensorNetworkState{V} <: AbstractTensorNetwork{V}
    tensornetwork::TensorNetwork{V}
    siteinds::Dictionary{V, Vector{<:Index}}
    index_order::Dictionary{V, Vector{<:Index}}
    index_directions::Dictionary{V, Vector{Bool}}
    grading::Dictionary{Index, Vector{Bool}}
end

tensornetwork(ftns::FermionicTensorNetworkState) = ftns.tensornetwork
siteinds(ftns::FermionicTensorNetworkState) = ftns.siteinds
graph(ftns::FermionicTensorNetworkState) = graph(tensornetwork(ftns))
tensors(ftns::FermionicTensorNetworkState) = tensors(tensornetwork(ftns))

index_order(ftns::FermionicTensorNetworkState) = ftns.index_order
index_directions(ftns::FermionicTensorNetworkState) = ftns.index_directions
grading(ftns::FermionicTensorNetworkState) = ftns.grading

# `prime` preserves an index's `id` (only `plev` changes), so the grading is shared between
# the ket layer and its primed bra-layer counterpart. The bra layer's global-order positions
# are derived from the ket positions by the uniform shift in `_effective_gpos` (primed indices
# pushed past all ket indices, preserving relative order).

function Base.copy(ftns::FermionicTensorNetworkState)
    return FermionicTensorNetworkState(
        copy(tensornetwork(ftns)), copy(siteinds(ftns)), copy(index_order(ftns)), copy(index_directions(ftns)), copy(grading(ftns))
    )
end

siteinds(ftns::FermionicTensorNetworkState, v) = siteinds(ftns)[v]

#Forward onto the underlying tensor network
for f in [:(Base.getindex), :add_tensor!, :(NamedGraphs.rem_vertex!)]
    @eval begin
        function $f(ftns::FermionicTensorNetworkState, args...; kwargs...)
            return $f(tensornetwork(ftns), args...; kwargs...)
        end
    end
end

"""
    _random_even_itensor(eltype, is::Vector{<:Index}, grading)

Build a random ITensor over indices `is` whose components with *odd* total Z2 parity are
zeroed, so the result is parity even with respect to `grading`.
"""
function _random_even_itensor(eltype, is::Vector{<:Index}, grading::Dictionary{Index, Vector{Bool}})
    bits = [grading[i] for i in is]
    dims = ntuple(k -> dim(is[k]), length(is))
    arr = zeros(eltype, dims...)
    for I in CartesianIndices(dims)
        odd = false
        for k in 1:length(is)
            odd ⊻= bits[k][I[k]]
        end
        odd || (arr[I] = randn(eltype))
    end
    return ITensor(arr, is...)
end

"""
    random_fermionic_tensornetworkstate(eltype, g::AbstractGraph; bond_dimension = 1, bond_grading = ...)

Generate a random parity-even `FermionicTensorNetworkState` on graph `g`.

Each vertex carries a single spinless-fermion site index (dimension 2, grading `[false, true]`
i.e. `|0⟩` even / `|1⟩` odd). Neighbouring tensors share one graded virtual (bond) index of
dimension `bond_dimension`, whose even/odd split is given by `bond_grading`. Every tensor is
made parity even by zeroing odd-total-parity components.

This is the locally-ordered formalism (arXiv:2410.02215): each tensor stores its own leg
`index_order`, and each shared bond carries an arrow via `index_directions` (`true` = in/−,
`false` = out/+). For every edge the `src` endpoint holds the bond as out and the `dst`
endpoint as in, so the arrow points `src → dst`. Site legs point out. Fermionic signs are
resolved per-bond at contraction time by [`fermionic_contract`](@ref) — there is no global
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

    grading = Dictionary{Index, Vector{Bool}}()
    index_order = Dictionary{vertextype(g), Vector{<:Index}}()
    index_directions = Dictionary{vertextype(g), Vector{Bool}}()

    # Site indices first
    for v in vs
        for sind in siteinds[v]
            if dim(sind) == 2 && occursin("fermion", string(tags(sind)))
                set!(grading, sind, [false, true])
            elseif dim(sind) == 4 && occursin("spinful_fermion", string(tags(sind)))
                set!(grading, sind, [false, true, true, false])
            else
                error("Don't recognize this as a fermionic site index")
            end
            set!(index_order, v, [sind])
            set!(index_directions, v, [false])
        end
    end

    # Each bond is appended to both endpoints' leg order, with an arrow `src → dst`
    # (src holds it as out/+, dst as in/−). Arrows are local to bonds, so no global
    # ordering of the tensors is needed (locally-ordered formalism, arXiv:2410.02215).
    l = Dict{NamedEdge{vertextype(g)}, Index}()
    for e in edges(g)
        b = Index(bond_dimension, "Fermion,Link")
        set!(grading, b, bond_grading)
        index_order[src(e)] = [index_order[src(e)]; [b]]
        index_order[dst(e)] = [index_order[dst(e)]; [b]]
        index_directions[src(e)] = [index_directions[src(e)]; [false]]
        index_directions[dst(e)] = [index_directions[dst(e)]; [true]]
        l[e] = b
        l[reverse(e)] = b
    end

    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        is = Index[siteinds[v]; [l[NamedEdge(v => vn)] for vn in neighbors(g, v)]]
        set!(tensors, v, _random_even_itensor(eltype, is, grading))
    end


    return FermionicTensorNetworkState(TensorNetwork(tensors, g), siteinds, index_order, index_directions, grading)
end

function random_fermionic_tensornetworkstate(g::AbstractGraph, args...; kwargs...)
    return random_fermionic_tensornetworkstate(Float64, g, args...; kwargs...)
end

"""
    FermionicTensor(ftns::FermionicTensorNetworkState, v)

Build the locally-ordered [`FermionicTensor`](@ref) for vertex `v`: its dense
tensor together with the stored leg order, arrow directions, and grading.
"""
function FermionicTensor(ftns::FermionicTensorNetworkState, v)
    return FermionicTensor(ftns[v], copy(index_order(ftns)[v]), copy(index_directions(ftns)[v]), grading(ftns))
end