# The operator layer is any tensor network on the same graph mapping the ket's site indices to
# their primed copies: a product of on-site operators (a `TensorNetworkState`, the default) or a
# `TensorNetworkOperator` with virtual bonds such as the generating operator of a Hamiltonian.
struct QuadraticForm{V, K <: TensorNetworkState{V}, O <: AbstractTensorNetwork{V}} <: AbstractForm{V}
    ket::K
    operator::O
end

ket(qf::QuadraticForm) = qf.ket
operator(qf::QuadraticForm) = qf.operator
#No whole-network `bra`: the dual is taken lazily per vertex/edge below, and there is no
#`prime`/`dag` for a TensorNetworkState to build one with.
# Dangling "Charge" legs (the root vertex of a charged graded state, see `charged` product states in
# kernel_hooks.jl) pair bra–ket directly, as `norm_factors` does: left primed on the bra they dangle
# in every region contraction, the messages of a loopy network carry them around and two messages
# meeting at a vertex both hold the same leg (measured: BP on the spinful fU1xU1 hexagon failed with
# "Contracted axes do not match" on the (−3,−3) root Charge leg; fZ2 was spared only because a
# six-electron state has trivial total parity and no Charge leg).
bra_tensor(qf::QuadraticForm, v) = unprime_charge_legs(dag(prime(ket(qf)[v])), ket(qf)[v])
bra_virtualinds(qf::QuadraticForm, edge::NamedEdge) = dag.(prime.(virtualinds(ket(qf), edge)))

Base.copy(qf::QuadraticForm) = QuadraticForm(copy(qf.ket), copy(qf.operator))

#Constructor, bra is taken to be in the vector space of ket so the dual is taken
function QuadraticForm(ket::TensorNetworkState, f::Function = v -> "I")
    sinds = siteinds(ket)
    verts = collect(vertices(ket))
    dtype = datatype(ket)
    operator_tensors = adapt(dtype).([reduce(*, [op(f(v), sind) for sind in sinds[v]]) for v in verts])
    operator = TensorNetworkState(Dictionary(verts, operator_tensors))
    return QuadraticForm(ket, operator)
end
