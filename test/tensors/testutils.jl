# Array of a Tensor in a requested index order.
function tarray(t::Tensor, is...)
    perm = map(i -> findfirst(==(i), t.inds), collect(is))
    return permutedims(t.data, perm)
end

# Dense Jordan-Wigner reference implementation shared by the fermionic tests.
const JW_ANNIHILATION = ComplexF64[0 1; 0 0]
const JW_PARITY = ComplexF64[1 0; 0 -1]
const JW_IDENTITY = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)

function jw_ops(n)
    return [
        reduce(kron, [
            k < j ? JW_PARITY : (k == j ? JW_ANNIHILATION : JW_IDENTITY) for k in 1:n
        ]) for j in 1:n
    ]
end

function jw_evolve(layer, cs, mode, n; occupied = ())
    ψv = zeros(ComplexF64, 2^n)
    ψv[1 + sum(v -> 2^(n - mode[v]), occupied; init = 0)] = 1.0
    return foldl(layer; init = ψv) do ϕv, gate
        if gate[1] == "F_phase"
            j = mode[only(gate[2])]
            exp(-im * gate[3] * (cs[j]' * cs[j])) * ϕv
        else
            j, k = mode[gate[2][1]], mode[gate[2][2]]
            H = gate[1] == "F_hop" ? (cs[j]' * cs[k] + cs[k]' * cs[j]) :
                (cs[j]' * cs[k]' + cs[k] * cs[j])
            exp(-im * gate[3] * H) * ϕv
        end
    end
end
