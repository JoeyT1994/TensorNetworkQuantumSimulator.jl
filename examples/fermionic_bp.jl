using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient

Random.seed!(1234)

println("Building Spinless Fermion Tensor Network on a 5 x 5 comb tree")
g = named_comb_tree((4,4))
s = siteinds("fermion", g)
ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)
ψ = normalize(ψ; alg = "bp")

println("Testing the norm")
@show norm_sqr(ψ; alg = "exact")
@show norm_sqr(ψ; alg = "bp")

println("Testing hopping CiCjdag + CjCidag on nearest neighbors")
e_exact = expect(ψ, (["Cdag", "C"], [(2,2), (2,3)]); alg = "exact") + expect(ψ, (["Cdag", "C"], [(2,3), (2,2)]); alg = "exact")
e_bp = expect(ψ, (["Cdag", "C"], [(2,2), (2,3)]); alg = "bp") + expect(ψ, (["Cdag", "C"], [(2,3), (2,2)]); alg = "bp")
println("Exact hopping is $e_exact")
println("BP hopping is $e_bp")

println("------------------")

println("Building Spinless Fermion Tensor Network on a 4 x 4 square grid")
g = named_grid((4,4))
s = siteinds("spinful_fermion", g)
ψ = random_fermionic_tensornetworkstate(Float64, g, s; bond_dimension = 2)
ψ = normalize(ψ; alg = "bp")

println("Testing the norm")
@show norm_sqr(ψ; alg = "exact")
@show norm_sqr(ψ; alg = "bp")

println("Testing hopping CiupCjupdag + CjupCupidag on nearest neighbors")
e_exact = expect(ψ, (["Cupdag", "Cup"], [(2,2), (2,3)]); alg = "exact") + expect(ψ, (["Cupdag", "Cup"], [(2,3), (2,2)]); alg = "exact")
e_bp = expect(ψ, (["Cupdag", "Cup"], [(2,2), (2,3)]); alg = "bp") + expect(ψ, (["Cupdag", "Cup"], [(2,3), (2,2)]); alg = "bp")
println("Exact hopping is $e_exact")
println("BP hopping is $e_bp")

println("------------------")


println("Building Spinful Fermion Tensor Network on a 3x3 hexagonal grid")
g = named_hexagonal_lattice_graph(3,3)
s = siteinds("spinful_fermion", g)
ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)
ψ = normalize(ψ; alg = "bp")

println("Testing the norm")
@show norm_sqr(ψ; alg = "exact")
@show norm_sqr(ψ; alg = "bp")

println("Testing hopping CiupCjupdag + CjupCupidag on nearest neighbors")
e_exact = expect(ψ, (["Cupdag", "Cup"], [(2,2), (2,3)]); alg = "exact") + expect(ψ, (["Cupdag", "Cup"], [(2,3), (2,2)]); alg = "exact")
e_bp = expect(ψ, (["Cupdag", "Cup"], [(2,2), (2,3)]); alg = "bp") + expect(ψ, (["Cupdag", "Cup"], [(2,3), (2,2)]); alg = "bp")
println("Exact hopping is $e_exact")
println("BP hopping is $e_bp")

ψp = fermionic_tensornetworkstate(ComplexF32, v-> isodd(sum(v)) ? "Up" : "Dn", g, s)

@show expect(ψp, [(["Nup"], [v]) for v in vertices(ψp)]; alg = "bp") + expect(ψp, [(["Ndn"], [v]) for v in vertices(ψp)]; alg = "bp")