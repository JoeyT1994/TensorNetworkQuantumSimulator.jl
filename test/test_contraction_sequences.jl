@eval module $(gensym())
using TensorNetworkQuantumSimulator: scalar, random_itensor, new_index, contract
using Random
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using OMEinsumContractionOrders: NestedEinsum, EinCode, getixsv, getiyv
using Test: @testset, @test

# Collect the leaf tensor-positions of a (possibly nested) contraction sequence.
collect_leaves!(acc, x::Integer) = push!(acc, Int(x))
collect_leaves!(acc, x) = (for y in x; collect_leaves!(acc, y); end; acc)

@testset "Contraction sequences (omeinsum backend)" begin
    Random.seed!(1234)

    # --- to_eincode: tensors -> (EinCode, size_dict). Tests the omeinsum-specific
    #     input conversion directly (tensors -> EinCode), so a silent fallback to another backend can't pass it.
    i, j, k = new_index(2), new_index(3), new_index(4)
    A = random_itensor(Float64, i, j)
    B = random_itensor(Float64, j, k)
    code, size_dict = TNQS.to_eincode([A, B])
    @test Set(Set.(getixsv(code))) == Set([Set([i, j]), Set([j, k])])  # per-tensor index sets
    @test Set(getiyv(code)) == Set([i, k])                             # open indices (j is contracted)
    @test size_dict == Dict(i => 2, j => 3, k => 4)

    # --- to_contraction_sequence: NestedEinsum -> nested tensor-position tree. Tests our
    #     converter on hand-built trees with known shapes (deterministic, exact).
    dummy = EinCode([[1, 2], [2, 3]], [1, 3])   # converter reads args/tensorindex, ignores eins content
    L(t) = NestedEinsum{Int}(t)                  # leaf at tensor position t
    node(args) = NestedEinsum(args, dummy)       # internal node (tensorindex = -1)
    @test TNQS.to_contraction_sequence(L(5)) == 5                                   # leaf -> bare Int
    @test TNQS.to_contraction_sequence(node([L(1), L(3)])) == [1, 3]
    @test TNQS.to_contraction_sequence(node([node([L(1), L(3)]), L(2)])) == [[1, 3], 2]

    # --- backend output is a complete, well-formed contraction tree (for each optimizer).
    g = named_grid((3, 3))
    tn = random_tensornetwork(Float64, g; bond_dimension = 2)
    tensors = [tn[v] for v in vertices(tn)]
    n = length(tensors)
    for optimizer in (GreedyMethod(), TreeSA(), ExhaustiveSearch())
        seq = TNQS.contraction_sequence(tensors; alg = "omeinsum", optimizer)
        @test sort(collect_leaves!(Int[], seq)) == collect(1:n)             # every tensor exactly once
        @test seq isa AbstractVector && any(x -> x isa AbstractVector, seq) # nested tree, not a flat list
    end

    # --- the sequence the backend returns is a *correct* contraction: executing it gives the
    #     same scalar as the independent `optimal` backend.
    ref = scalar(contract(tensors; sequence = TNQS.contraction_sequence(tensors; alg = "optimal")))
    for optimizer in (GreedyMethod(), TreeSA(), ExhaustiveSearch())
        seq = TNQS.contraction_sequence(tensors; alg = "omeinsum", optimizer)
        @test scalar(contract(tensors; sequence = seq)) ≈ ref
    end

    # --- open network: result is a tensor with dangling indices (iy non-empty).
    p, q, r, s, t = new_index(2), new_index(3), new_index(2), new_index(3), new_index(2)
    X = random_itensor(Float64, p, q)
    Y = random_itensor(Float64, q, r, s)
    Z = random_itensor(Float64, s, t)
    open_tensors = [X, Y, Z]   # open indices: p, r, t
    seq_open = TNQS.contraction_sequence(open_tensors; alg = "omeinsum", optimizer = GreedyMethod())
    @test sort(collect_leaves!(Int[], seq_open)) == [1, 2, 3]
    @test contract(open_tensors; sequence = seq_open) ≈
        contract(open_tensors; sequence = TNQS.contraction_sequence(open_tensors; alg = "optimal"))
end
end
