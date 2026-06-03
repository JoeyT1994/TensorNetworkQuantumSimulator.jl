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

# Fermionic analogue of the bosonic `norm_factors` (src/TensorNetworks/tensornetworkstate.jl):
# for each vertex emit its ket tensor, its bra tensor, and any single-site operator factor,
# producing the flat `2N + n_ops` parity-even tensor list for ⟨ψ|O|ψ⟩ (unnormalised). The
# caller folds the list with the order-independent `contract`; the optimal sequence finder
# picks the fold tree. `op_strings(v)` gives the operator name on vertex `v` ("I"/"ρ" =
# identity). Returns `nothing` when `O` has odd total parity (⟨O⟩ = 0).
function fermionic_norm_factors(ψ::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I")
    # Pre-scan odd operators: an odd number of odd factors is parity-forbidden, and a pair
    # shares one dummy odd "operator-string" bond `d` (Eq.95-96) created once here.
    odd_vs = [v for v in verts if is_odd(op_strings(v))]
    isodd(length(odd_vs)) && return nothing             # parity-forbidden ⇒ ⟨O⟩ = 0
    length(odd_vs) > 2 && error("Fermionic expect currently supports at most one pair of odd operators (e.g. a single hopping term).")
    d = isempty(odd_vs) ? nothing : Index(1, "Fermion,OpString")

    factors = FermionicITensor[]
    for v in verts
        name = op_strings(v)
        op_here = !(name == "I" || name == "ρ")
        ket = ψ[v]
        # keep the un-operated site legs unprimed; prime everything else (bonds, op sites)
        keep = op_here ? Index[] : collect(siteinds(ψ, v))
        push!(factors, ket, dag(ket, keep))
        op_here || continue
        s = only(siteinds(ψ, v))
        (dim(s) == 2 || dim(s) == 4) || error("Fermionic measurement currently supports spinless (dimension-2) or spinful (dimension-4) sites only.")
        sgr = ket.grading[s]
        # A pair of ODD factors (e.g. a hopping c_i† c_j) shares the dummy `d`; each carries
        # it with the arrow set by `odd_op_tensor` (creation = bra, annihilation = ket), so
        # `contract` inserts the supertrace g once — supplying the (−1) of the fermionic
        # operator ordering automatically, regardless of fold order.
        push!(factors, is_odd(name) ? odd_op_tensor(s, name, d, sgr) : even_op_tensor(s, name, sgr))
    end
    return factors
end