@eval module $(gensym())
using Test: @test, @testset, @test_throws
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator.Tensors: Tensors, Index, Tensor
using Graphs: Graphs
using NamedGraphs: NamedGraph
const TI = TensorNetworkQuantumSimulator.TensorInterface
using LinearAlgebra: LinearAlgebra, norm, qr, factorize
using Random: Random

include("tensors/testutils.jl")

# Set TNQS_TENSOR_TEST_PARTS to a comma-separated subset for focused runs, for
# example `dense,bp` or `fermionic-chain`. The default remains the complete suite.
const TENSOR_TEST_PARTS = (
    ("dense", "tensors/test_dense.jl"),
    ("graded-z2", "tensors/test_graded_z2.jl"),
    ("bp", "tensors/test_bp.jl"),
    ("fermionic-chain", "tensors/test_fermionic_chain.jl"),
    ("fermionic-spinful", "tensors/test_fermionic_spinful.jl"),
    ("fermionic-2d", "tensors/test_fermionic_2d.jl"),
)

function selected_tensor_test_parts()
    known = Set(first.(TENSOR_TEST_PARTS))
    raw = strip(get(ENV, "TNQS_TENSOR_TEST_PARTS", "all"))
    lowercase(raw) == "all" && return known

    requested = Set(filter(x -> !isempty(x), strip.(split(raw, ','))))
    isempty(requested) && error("TNQS_TENSOR_TEST_PARTS must be `all` or a comma-separated list")
    unknown = setdiff(requested, known)
    isempty(unknown) || error(
        "unknown TNQS_TENSOR_TEST_PARTS value(s): $(join(sort!(collect(unknown)), ", ")). " *
            "Known parts: $(join(sort!(collect(known)), ", "))"
    )
    return requested
end

@testset "Tensors backend" begin
    selected = selected_tensor_test_parts()
    for (part, file) in TENSOR_TEST_PARTS
        part in selected || continue
        @info "Starting tensor tests" part
        elapsed = @elapsed include(file)
        @info "Finished tensor tests" part seconds = round(elapsed; digits = 2)
    end
end
end
