struct BilinearForm{V} <: AbstractForm{V}
    ket::TensorNetworkState{V}
    operator::TensorNetworkState{V}
    bra::TensorNetworkState{V}
end

ket(blf::BilinearForm) = blf.ket
operator(blf::BilinearForm) = blf.operator
bra(blf::BilinearForm) = blf.bra
bra_tensor(blf::BilinearForm, v) = bra(blf)[v]
bra_virtualinds(blf::BilinearForm, edge::NamedEdge) = virtualinds(bra(blf), edge)
bra_factor_inds(blf::BilinearForm, v) = factor_inds(bra(blf), v)

Base.copy(blf::BilinearForm) = BilinearForm(copy(blf.ket), copy(blf.operator), copy(blf.bra))

#Constructor, bra is taken to be in the vector space of ket so the dual is taken
#
# `consume_bra` swaps the duplicate for an in-place transform: `prime` shares storage, so
# `dag(prime(·))` costs exactly one conjugation, and doing it in place allocates nothing at all.
# The caller's `bra` is destroyed in exchange. Worth having because the out-of-place default
# leaves the original and the conjugate both resident -- 15 GiB apiece at χ=800 with S=4 -- for
# the whole lifetime of the form, and a caller computing `inner(ψ, ϕ)` usually has no further use
# for `ϕ`. See [`dag_prime!`](@ref).
function BilinearForm(
        ket::TensorNetworkState, bra::TensorNetworkState; consume_bra::Bool = false
    )
    dtype = datatype(ket)
    @assert graph(ket) == graph(bra)
    # `protect = ket`: a bra derived from the ket shares storage with it at every vertex no
    # gate touched, and conjugating those in place would corrupt the ket.
    bra = consume_bra ? dag_prime!(bra; protect = ket) : map_tensors(t -> dag(prime(t)), bra)
    sinds = siteinds(ket)
    verts = collect(vertices(ket))
    operator_tensors = [adapt(dtype)(reduce(*, ITensor[denseblocks(delta(sind, prime(dag(sind)))) for sind in sinds[v]])) for v in verts]
    operator = TensorNetworkState(Dictionary(verts, operator_tensors))
    return BilinearForm(ket, operator, bra)
end
