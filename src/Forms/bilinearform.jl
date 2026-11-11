struct BilinearForm{T, V, K <: ITensorNetwork{<:Any, V}, B <: ITensorNetwork{<:Any, V}} <: AbstractForm{T, V}
    ket::K
    bra::B
end

ITNN.ketnetwork(blf::BilinearForm) = blf.ket
ITNN.branetwork(blf::BilinearForm) = blf.bra
ITNN.factor_tensors(blf::BilinearForm, v) = [blf.ket[v], blf.bra[v]]
# The bra is stored already conjugated, so no operand is flagged for conjugation.
function ITNN.message_contraction_tensors(blf::BilinearForm, messages, v)
    return [blf.ket[v]; messages...; blf.bra[v]], falses(length(messages) + 2)
end

Base.copy(blf::BilinearForm) = typeof(blf)(copy(blf.ket), copy(blf.bra))

#Constructor, bra is taken to be in the vector space of ket so the dual is taken
function BilinearForm(ket::ITensorNetwork{T, V}, bra::ITensorNetwork) where {T, V}
    graph(ket) == graph(bra) || error("BilinearForm: ket and bra must share the same graph")
    nn = ITNN.normnetwork(ITensorBase.prime, bra)
    verts = collect(vertices(bra))
    bra = ITensorNetwork(Dictionary(verts, [ITNN.bratensor(nn, v) for v in verts]))
    return BilinearForm{T, V, typeof(ket), typeof(bra)}(ket, bra)
end

#scalar/storage types must span both stored networks (the generic AbstractForm methods
#forward to the ket only, which under-reports for an independently-typed bra)
scalartype(blf::BilinearForm) = promote_type(scalartype(blf.ket), scalartype(blf.bra))
