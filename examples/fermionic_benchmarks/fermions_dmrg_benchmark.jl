using TensorNetworkQuantumSimulator
using ITensorMPS

using ITensorMPS: ITensorMPS


function main()

    χ = 200
    t_parr, t_perp= 1.0, 0.5
    U = 8
    n =10
    g = named_grid((n,2); periodic = false)
    nv = length(vertices(g))
    s = ITensorMPS.siteinds("Electron", nv; conserve_qns = true)

    state = [isodd(sum(v)) ? "Up" : "Dn" for v in vertices(g)]
    ψ = randomMPS(s, state, χ)

    ordered_vs = reduce(vcat, [[(i, 1), (i,2)] for i in 1:n])
    # Build the Hamiltonian
    os = OpSum()
    for e in edges(g)
        i,j = findfirst(v -> v == src(e), ordered_vs), findfirst(v -> v == dst(e), ordered_vs)
        t = src(e)[1] == dst(e)[1] ? t_perp : t_parr
        os += -t, "Cdagup", i, "Cup", j
        os += -t, "Cdagup", j, "Cup", i
        os += -t, "Cdagdn", i, "Cdn", j
        os += -t, "Cdagdn", j, "Cdn", i
    end
    for vp in vertices(g)
        i = findfirst(v -> v == vp, ordered_vs)
        os += U, "Nup", i, "Ndn", i
        os += -U/2, "Ntot", i
    end


    # Convert algebraic sum to MPO
    H = MPO(os, s)

      # Run DMRG to find the ground state
    sweeps = Sweeps(10)
    maxdim!(sweeps, χ)
    cutoff!(sweeps, 1e-10)
    energy, ψ = dmrg(H, ψ, sweeps)

    println("Total energy post DMRG: $energy")


end

main()