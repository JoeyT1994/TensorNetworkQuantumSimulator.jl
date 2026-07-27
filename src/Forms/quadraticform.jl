struct QuadraticForm{V} <: AbstractForm{V}
    ket::TensorNetworkState{V}
    operator::TensorNetworkState{V}
end

ket(qf::QuadraticForm) = qf.ket
operator(qf::QuadraticForm) = qf.operator
#No whole-network `bra`: the dual is taken lazily per vertex/edge below, and there is no
#`prime`/`dag` for a TensorNetworkState to build one with.
bra_tensor(qf::QuadraticForm, v) = dag(prime(ket(qf)[v]))
bra_virtualinds(qf::QuadraticForm, edge::NamedEdge) = dag.(prime.(virtualinds(ket(qf), edge)))

Base.copy(qf::QuadraticForm) = QuadraticForm(copy(qf.ket), copy(qf.operator))

#Constructor, bra is taken to be in the vector space of ket so the dual is taken
function QuadraticForm(ket::TensorNetworkState, f::Function = v -> "I")
    sinds = siteinds(ket)
    verts = collect(vertices(ket))
    dtype = datatype(ket)
    operator_tensors = adapt(dtype).([reduce(*, ITensor[ITensors.op(f(v), sind) for sind in sinds[v]]) for v in verts])
    operator = TensorNetworkState(Dictionary(verts, operator_tensors))
    return QuadraticForm(ket, operator)
end
