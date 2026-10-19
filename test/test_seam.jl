@eval module $(gensym())
using TensorNetworkQuantumSimulator
using Test: @testset, @test

# The backend seam (src/tensor_interface.jl, rule 1): no source file outside src/Backends/
# references a tensor library or the backend module. Comments and docstrings may name them as
# a reference for conventions; code may not.

const FORBIDDEN = r"\b(ITensors|NDTensors|ITensorBase|GradedArrays|TensorAlgebra|TensorKit\w*|MatrixAlgebraKit|ITensorsBackend|ITensorsCompat)\b"

# The source with block comments, docstrings and line comments removed (line numbers kept).
function strip_comments(src::AbstractString)
    blank(m) = "\n"^count(==('\n'), m)
    src = replace(src, r"#=.*?=#"s => blank)
    src = replace(src, r"\"\"\".*?\"\"\""s => blank)
    return join((replace(l, r"#.*$" => "") for l in split(src, '\n')), '\n')
end

@testset "Seam: no tensor library outside src/Backends" begin
    root = joinpath(pkgdir(TensorNetworkQuantumSimulator), "src")
    files = String[]
    for (dir, _, fs) in walkdir(root), f in fs
        endswith(f, ".jl") && push!(files, joinpath(dir, f))
    end
    core = filter(p -> first(splitpath(relpath(p, root))) != "Backends", files)
    @test length(core) > 20
    offending = String[]
    for path in core
        for (n, line) in enumerate(split(strip_comments(read(path, String)), '\n'))
            occursin(FORBIDDEN, line) && push!(offending, "$(relpath(path, root)):$n: $(strip(line))")
        end
    end
    isempty(offending) || println("Tensor-library references outside src/Backends:\n", join(offending, "\n"))
    @test isempty(offending)
    # the stripping itself: a forbidden name in code is caught, in comments it is not
    @test occursin(FORBIDDEN, strip_comments("x = ITensors.op(\"X\", s) # comment"))
    @test !occursin(FORBIDDEN, strip_comments("#= ITensors =#\n\"\"\"\nITensors\n\"\"\"\nx = 1 # ITensors"))
end

end
