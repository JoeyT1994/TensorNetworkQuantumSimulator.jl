@eval module $(gensym())
using ITensors: ITensors
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: GATES, OPERATORS, STATES, toitensor, op, state, register_op!
using Test: @testset, @test, @test_throws

# The in-house operator library against the ITensors.op it replaced. Names ITensors defines are
# compared with ITensors.op; the in-house gates, whose ITensors.op overloads lived in this package,
# with the formulas of those overloads.

const θ, ϕ, β, λ = 0.37, -0.81, 0.23, 0.59

# Keyword arguments each parametric operator takes (as ITensors names them).
function op_kwargs(name)
    name in ("Rx", "Ry", "Rz", "CRx", "CRy", "CRz", "CRX", "CRY", "CRZ", "Rz+", "Rz+z+", "Rxxyy", "Rxxyyzz") && return (; θ)
    name in ("Phase", "P", "CPHASE", "Cphase", "Rxx", "Ryy", "Rzz", "Rxy", "RXX", "RYY", "RZZ", "RXY") && return (; ϕ)
    name == "xx_plus_yy" && return (; θ, β)
    return (;)
end

# Matrix on (s1, s2) with the first site most significant, as ITensors lays out its op matrices.
ref_matrix(M, s1, s2) = ITensors.itensor(M, ITensors.prime(s2), ITensors.prime(s1), s2, s1)
ref_matrix(M, s1) = ITensors.itensor(M, ITensors.prime(s1), s1)

function inhouse_reference(name, sites; kw...)
    X(s) = ITensors.op("X", s); Y(s) = ITensors.op("Y", s); Z(s) = ITensors.op("Z", s)
    if name == "Rxxyy"
        s1, s2 = sites
        return exp(-im * kw[:θ] * 0.5 * (X(s1) * X(s2) + Y(s1) * Y(s2)))
    elseif name == "Rxxyyzz"
        s1, s2 = sites
        return exp(-im * kw[:θ] * 0.5 * (X(s1) * X(s2) + Y(s1) * Y(s2) + Z(s1) * Z(s2)))
    elseif name == "xx_plus_yy"
        t, b = kw[:θ], kw[:β]
        M = [
            1 0 0 0
            0 cos(t / 2) -im * sin(t / 2) * exp(-im * b) 0
            0 -im * sin(t / 2) * exp(im * b) cos(t / 2) 0
            0 0 0 1
        ]
        return ref_matrix(M, sites...)
    elseif name == "Rz+"
        return ref_matrix([1 0; 0 exp(-im * kw[:θ] * 0.5)], sites...)
    elseif name == "Rz+z+"
        return ref_matrix([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(-im * kw[:θ] * 0.5)], sites...)
    end
    return nothing
end

function reference(name, sites; kw...)
    r = inhouse_reference(name, sites; kw...)
    return isnothing(r) ? ITensors.op(name, sites...; kw...) : r
end

same(a, b) = ITensors.inds(a) ⊆ ITensors.inds(b) && ITensors.inds(b) ⊆ ITensors.inds(a) && eltype(a) == eltype(b) && a ≈ b

@testset "Operator library matches ITensors.op" begin
    g = named_path_graph(2)
    s = siteinds("S=1/2", g)
    s1, s2 = only(s[1]), only(s[2])

    @testset "Every registered gate, through the circuit-tuple path" begin
        for (name, spec) in GATES
            nsites = name in ("X", "Y", "Z", "H", "Rx", "Ry", "Rz", "P", "Rz+") ? 1 : 2
            verts = nsites == 1 ? [1] : [1, 2]
            sites = nsites == 1 ? [s1] : [s1, s2]
            param = length(spec.paramkeys) == 2 ? (θ, β) : θ
            gate = isempty(spec.paramkeys) ? (name, verts) : (name, verts, param)
            t, vs = toitensor(gate, g, s)
            @test vs == verts
            raw = isempty(spec.paramkeys) ? () : spec.rescale(param)
            pvals = raw isa Tuple ? raw : (raw,)
            kw = NamedTuple{spec.paramkeys}(pvals)
            @test same(t, reference(spec.opname, sites; kw...))
        end
    end

    @testset "Registry operators on S=1/2 and S=1 sites" begin
        q = ITensors.Index(3, "S=1")
        for ((nsites, d), table) in OPERATORS, name in keys(table)
            sites = d == 2 ? (nsites == 1 ? [s1] : [s1, s2]) : [q]
            kw = op_kwargs(name)
            @test same(op(name, sites...; kw...), reference(name, sites; kw...))
        end
    end

    @testset "States" begin
        q = ITensors.Index(3, "S=1")
        for (d, table) in STATES, name in keys(table)
            site = d == 2 ? s1 : q
            @test same(state(name, site), ITensors.state(name, site))
        end
    end

    @testset "Fallback to the backend library, and errors" begin
        # "Rn" is defined by ITensors but not by the package registry
        @test same(op("Rn", s1; θ, ϕ, λ), ITensors.op("Rn", s1; θ, ϕ, λ))
        @test_throws ArgumentError op("NotAnOperator", s1)
        # and the other way: the in-house gates stay reachable as ITensors.op (ITensorsCompat)
        for name in ("Rxxyy", "Rxxyyzz", "xx_plus_yy")
            kw = op_kwargs(name)
            @test same(ITensors.op(name, s1, s2; kw...), op(name, s1, s2; kw...))
        end
        @test_throws ArgumentError state("NotAState", s1)
    end

    @testset "register_op!" begin
        register_op!("MyHalfZ", [0.5 0.0; 0.0 -0.5])
        @test same(op("MyHalfZ", s1), ITensors.op("Sz", s1))
        register_op!("MyRot", (; θ) -> [cos(θ) -sin(θ); sin(θ) cos(θ)])
        @test same(op("MyRot", s1; θ = π / 2), ref_matrix([0.0 -1.0; 1.0 0.0], s1))
        register_op!("MySWAP", [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]; nsites = 2)
        @test same(op("MySWAP", s1, s2), ITensors.op("SWAP", s1, s2))
    end
end

end
