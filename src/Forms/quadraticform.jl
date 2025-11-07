struct QuadraticForm{V} <: AbstractTensorNetwork{V}
    ket::TensorNetworkState{V}
    operator::TensorNetworkState{V}
end

ket(qf::QuadraticForm) = qf.ket
operator(qf::QuadraticForm) = qf.operator
bra(qf::QuadraticForm) = prime(dag(ket(qf)))

Base.copy(qf::QuadraticForm) = QuadraticForm(copy(qf.ket), copy(qf.operator))

#Forward onto the ket
for f in [
        :(graph),
        :(ITensors.datatype),
        :(ITensors.NDTensors.scalartype),
        :(NamedGraphs.edgeinduced_subgraphs_no_leaves),
    ]
    @eval begin
        function $f(qf::QuadraticForm, args...; kwargs...)
            return $f(ket(qf), args...; kwargs...)
        end
    end
end

#Constructor, bra is taken to be in the vector space of ket so the dual is taken
function QuadraticForm(ket::TensorNetworkState, f::Function = v -> "I")
    sinds = siteinds(ket)
    verts = collect(vertices(ket))
    dtype = datatype(ket)
    operator_tensors = adapt(dtype).([reduce(prod, ITensor[ITensors.op(f(v), sind) for sind in sinds[v]]) for v in verts])
    operator = TensorNetworkState(Dictionary(verts, operator_tensors))
    return QuadraticForm(ket, operator)
end

function default_message(qf::QuadraticForm, edge::AbstractEdge)
    linds = virtualinds(qf, edge)
    return adapt(datatype(qf))(denseblocks(delta(linds)))
end

function virtualinds(qf::QuadraticForm, edge::NamedEdge)
    ket_linds = virtualinds(ket(qf), edge)
    return Index[ket_linds; virtualinds(operator(qf), edge); dag.(prime.(ket_linds))]
end

function bp_factors(qf::QuadraticForm, verts::Vector)
    factors = ITensor[]
    for v in verts
        qf_v = ket(qf)[v]
        append!(factors, ITensor[qf_v, operator(qf)[v], dag(prime(qf_v))])
    end
    return factors
end

bp_factors(qf::QuadraticForm, v) = bp_factors(qf, [v])
