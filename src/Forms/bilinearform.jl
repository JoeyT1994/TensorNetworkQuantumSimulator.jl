struct BilinearForm{T, V, K <: ITensorNetwork{<:Any, V}, B <: ITensorNetwork{<:Any, V}} <: AbstractForm{T, V}
    ket::K
    bra::B
end

ITNN.ketnetwork(blf::BilinearForm) = blf.ket
ITNN.branetwork(blf::BilinearForm) = blf.bra
ITNN.factor_tensors(blf::BilinearForm, v) = [blf.ket[v], blf.bra[v]]

Base.copy(blf::BilinearForm) = typeof(blf)(copy(blf.ket), copy(blf.bra))

#Constructor, bra is taken to be in the vector space of ket so the dual is taken
function BilinearForm(ket::ITensorNetwork{T, V}, bra::ITensorNetwork) where {T, V}
    @assert graph(ket) == graph(bra)
    nn = ITNN.normnetwork(prime, bra)
    verts = collect(vertices(bra))
    bra = ITensorNetwork(Dictionary(verts, [ITNN.bratensor(nn, v) for v in verts]))
    return BilinearForm{T, V, typeof(ket), typeof(bra)}(ket, bra)
end
