# Compare :cut, :cycle, and boundary MPS on the saved 9x9 D=3 Ising PEPS.
# Generate the input with `python examples/export_peps.py 9x9`.
using TensorNetworkQuantumSimulator
using ITensors
using Dictionaries: Dictionary, set!
using LinearAlgebra: BLAS
using NamedGraphs: NamedEdge
using Printf
using Statistics: mean
const TNQS = TensorNetworkQuantumSimulator

# The non-Hermitian cycle Schur solve can select a different near-degenerate branch under
# multithreaded reduction ordering. Keep benchmark data reproducible; callers can override.
BLAS.set_num_threads(parse(Int, get(ENV, "ISING9_BLAS_THREADS", "1")))

const NX = 9
const NY = 9
const BONDDIM = parse(Int, get(ENV, "ISING9_BOND_DIMENSION", "3"))
const BIN = joinpath(get(ENV, "CTM_ISING9X9_DIR", @__DIR__), "peps9x9_D$(BONDDIM).bin")
const ISING9_DEFAULT_F_REFERENCE = -38.024120943923315
const ISING9_DEFAULT_X_REFERENCE = 0.9277470295539833
rawshape(x,y) = (2, x==1 ? 1 : BONDDIM, x==NX ? 1 : BONDDIM,
                 y==1 ? 1 : BONDDIM, y==NY ? 1 : BONDDIM)

function load_peps()
    isfile(BIN) || error("Missing $BIN; run `python examples/export_peps.py 9x9` first.")
    data = open(BIN,"r") do f
        read!(f, Vector{Float64}(undef, filesize(BIN)÷8))
    end
    off=0; arrs=Dict{Tuple{Int,Int},Array{Float64,5}}()
    for x in 1:NX, y in 1:NY
        sh=rawshape(x,y); n=prod(sh); v=@view data[off+1:off+n]
        arrs[(x,y)] = permutedims(reshape(collect(v),reverse(sh)),(5,4,3,2,1)); off += n
    end
    @assert off==length(data)
    arrs
end

function build_state(arrs)
    g=named_grid((NX,NY)); s=siteinds("S=1/2",g); link=Dict{Any,Index}()
    for e in edges(g); link[e]=Index(BONDDIM,"Link"); link[reverse(e)]=link[e]; end
    tensors=Dictionary{Tuple{Int,Int},ITensor}()
    for x in 1:NX, y in 1:NY
        a=arrs[(x,y)]; v=(x,y); nbrs=((x-1,y),(x+1,y),(x,y-1),(x,y+1))
        keep=[i for i in 1:4 if size(a,i+1)>1]; idx=Index[only(s[v])]
        for i in keep; push!(idx,link[NamedEdge(v=>nbrs[i])]); end
        set!(tensors,v,ITensor(reshape(a,(2,[size(a,i+1) for i in keep]...)),idx))
    end
    TNQS.TensorNetworkState(TNQS.TensorNetwork(tensors,g),s)
end

parse_ints(s)=parse.(Int,split(s,','))
parse_bool(s)=lowercase(s) in ("1","true","yes","on")
parse_methods(s)=Symbol.(strip.(split(s,',')))
function bmps_values(ψ,χ,sites)
    cache=update(BoundaryMPSCache(ψ,χ; partition_by="row",gauge_state=false))
    lnN=log(abs(real(norm_sqr(cache;alg="boundarymps"))))
    cache=TNQS.update_partitions(cache,sites)
    observables=[("X",[site]) for site in sites]
    mx=mean(real.(expect(cache,observables;alg="boundarymps",
                         bmps_messages_up_to_date=true)))
    lnN,mx
end

function main()
    ψ=build_state(load_peps()); sites=collect(vertices(graph(ψ)))
    Fref=parse(Float64,get(ENV,"ISING9_REFERENCE_F",string(ISING9_DEFAULT_F_REFERENCE)))
    Xref=parse(Float64,get(ENV,"ISING9_REFERENCE_X",string(ISING9_DEFAULT_X_REFERENCE)))
    if parse_bool(get(ENV,"ISING9_RECOMPUTE_REFERENCE","false"))
        refχ=parse(Int,get(ENV,"ISING9_REF_CHI","96"))
        Fref,Xref=bmps_values(ψ,refχ,sites)
        @printf("# reference recomputed with bMPS chi=%d\n",refχ)
    end
    @printf("# 9x9 D=%d Ising PEPS; observable=mean X over %d sites\n",
            BONDDIM,length(sites))
    @printf("# F_ref=%.17g Xmean_ref=%.17g\n",Fref,Xref)
    get(ENV,"ISING9_REFERENCE_ONLY","false")=="true" && return
    @printf("chi,method,F,F_abs_error,X,X_abs_error,marginal_inconsistency,seconds\n")
    cycsub=parse_bool(get(ENV,"ISING9_CYCLE_SUBSPACE","false"))
    cycit=parse(Int,get(ENV,"ISING9_CYCLE_ITERS","20"))
    cycwarm=parse_bool(get(ENV,"ISING9_CYCLE_WARMSTART","true"))
    methods=parse_methods(get(ENV,"ISING9_METHODS","cut,cycle,bmps"))
    for χ in parse_ints(get(ENV,"ISING9_CHIS","2,4,6,8,10,12,14,16"))
        for proj in (:cut,:cycle)
            proj in methods || continue
            t=@elapsed c=update(CTMEnvironmentCache(ψ,χ;projector=proj,
                cycle_subspace=(proj==:cycle && cycsub),cycle_iters=cycit,cycle_warmstart=cycwarm);
                convergence=:environment,tolerance=1e-10,maxiter=60)
            observables=[("X",[site]) for site in sites]
            F=cvm_freenergy(c); X=mean(real.(expect(c,observables)))
            @printf("%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.6f\n",χ,string(proj),F,abs(F-Fref),X,abs(X-Xref),marginal_inconsistency(c),t)
            flush(stdout)
        end
        if :bmps in methods
            t=@elapsed F,X=bmps_values(ψ,χ,sites)
            @printf("%d,bmps,%.17g,%.17g,%.17g,%.17g,NaN,%.6f\n",χ,F,abs(F-Fref),X,abs(X-Xref),t)
            flush(stdout)
        end
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
