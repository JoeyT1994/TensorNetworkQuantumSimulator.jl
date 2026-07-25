# SHARED, position-resolved CTM environments, grown (no exact blocks) -> CVM free energy.
#
# Interface projector families (each a NESTED chain of eig isometries, derived ONCE):
#   PHN[x,y] : {hb[(x,j)] : j < y}   nested in y   (shared: CNW.right, CNE.left, TN.left/right)
#   PHS[x,y] : {hb[(x,j)] : j >= y}  nested down   (shared: CSW.right, CSE.left, TS.left/right)
#   PVW[x,y] : {vb[(i,y)] : i < x}   nested in x   (shared: CNW.down, CSW.up, TW.up/down)
#   PVE[x,y] : {vb[(i,y)] : i >= x}  nested down   (shared: CNE.down, CSE.up, TE.up/down)
# Blocks by DP growth:
#   TW[x,y]=cols<x,row y   TE[x,y]=cols>=x,row y   TN[x,y]=col x,rows<y   TS[x,y]=col x,rows>=y
#   CNW[x,y]=cols<x,rows<y   CNE[x,y]=cols>=x,rows<y   CSW[x,y]=cols<x,rows>=y   CSE=...
using ITensors, LinearAlgebra, Printf, Random

function rand_net(Lx,Ly,D;seed=7)
    Random.seed!(seed)
    hb=Dict{Tuple{Int,Int},Index}(); vb=Dict{Tuple{Int,Int},Index}()
    for y in 1:Ly, x in 1:Lx-1; hb[(x,y)]=Index(D,"h"); end
    for y in 1:Ly-1, x in 1:Lx; vb[(x,y)]=Index(D,"v"); end
    A=Dict{Tuple{Int,Int},ITensor}()
    for x in 1:Lx, y in 1:Ly
        is=Index[]; x<Lx&&push!(is,hb[(x,y)]); x>1&&push!(is,hb[(x-1,y)])
        y<Ly&&push!(is,vb[(x,y)]); y>1&&push!(is,vb[(x,y-1)])
        A[(x,y)]=ITensor(abs.(rand(ntuple(_->D,length(is))...)),is...)
    end
    return A,hb,vb
end

# isometry truncating index list `ins` of B to χ; returns (P,w), P legs (ins...,w)
function derive_proj(B, ins, χ)
    co=combiner(ins...); io=combinedind(co); d=ITensors.dim(io); k=min(χ,d)
    if k==d
        w=Index(d); return ITensor(Matrix{Float64}(I,d,d),io,w)*co, w
    end
    Bc=B*co; ρ=Bc*prime(dag(Bc),io)
    ρm=Array(ρ,io,prime(io)); F=eigen(Hermitian((ρm+ρm')/2))
    ord=sortperm(F.values;rev=true)[1:k]; w=Index(k)
    return ITensor(F.vectors[:,ord],io,w)*co, w
end

nn(d,k) = get(d,k,nothing)
mul(a,b) = isnothing(a) ? b : (isnothing(b) ? a : a*b)

function build_env(A,hb,vb,Lx,Ly,χ)
    TW=Dict{Tuple{Int,Int},Any}(); TE=Dict{Tuple{Int,Int},Any}()
    TN=Dict{Tuple{Int,Int},Any}(); TS=Dict{Tuple{Int,Int},Any}()
    CNW=Dict{Tuple{Int,Int},Any}(); CNE=Dict{Tuple{Int,Int},Any}()
    CSW=Dict{Tuple{Int,Int},Any}(); CSE=Dict{Tuple{Int,Int},Any}()
    PHN=Dict{Tuple{Int,Int},Tuple{ITensor,Index}}(); PHS=Dict{Tuple{Int,Int},Tuple{ITensor,Index}}()
    PVW=Dict{Tuple{Int,Int},Tuple{ITensor,Index}}(); PVE=Dict{Tuple{Int,Int},Tuple{ITensor,Index}}()

    # ---- W strips (x increasing, y increasing): derive PVW ----
    for y in 1:Ly
        for x in 1:Lx-1
            raw = mul(nn(TW,(x,y)), A[(x,y)])
            if y>1; raw = raw*PVW[(x+1,y-1)][1]; end
            if y<Ly
                ins=Index[]; x>=2 && push!(ins,PVW[(x,y)][2]); push!(ins,vb[(x,y)])
                P,w=derive_proj(raw,ins,χ); PVW[(x+1,y)]=(P,w); raw=raw*P
            end
            TW[(x+1,y)]=raw
        end
    end
    # ---- E strips (x decreasing): derive PVE ----
    for y in 1:Ly
        for x in Lx:-1:2
            raw = mul(A[(x,y)], nn(TE,(x+1,y)))
            if y>1; raw = raw*PVE[(x,y-1)][1]; end
            if y<Ly
                ins=Index[vb[(x,y)]]; x<=Lx-1 && push!(ins,PVE[(x+1,y)][2])
                P,w=derive_proj(raw,ins,χ); PVE[(x,y)]=(P,w); raw=raw*P
            end
            TE[(x,y)]=raw
        end
    end
    # ---- CNW (y increasing): derive PHN ----
    for x in 2:Lx
        for y in 1:Ly-1
            raw = mul(nn(CNW,(x,y)), TW[(x,y)])
            ins=Index[]; y>=2 && push!(ins,PHN[(x-1,y)][2]); push!(ins,hb[(x-1,y)])
            P,w=derive_proj(raw,ins,χ); PHN[(x-1,y+1)]=(P,w)
            CNW[(x,y+1)]=raw*P
        end
    end
    # ---- CSW (y decreasing): derive PHS ----
    for x in 2:Lx
        for y in Ly:-1:2
            raw = mul(nn(CSW,(x,y+1)), TW[(x,y)])
            ins=Index[hb[(x-1,y)]]; y<=Ly-1 && push!(ins,PHS[(x-1,y+1)][2])
            P,w=derive_proj(raw,ins,χ); PHS[(x-1,y)]=(P,w)
            CSW[(x,y)]=raw*P
        end
    end
    # ---- CNE / CSE: consume PHN / PHS ----
    for x in 2:Lx
        for y in 1:Ly-1
            raw = mul(nn(CNE,(x,y)), TE[(x,y)])
            CNE[(x,y+1)] = raw*dag(PHN[(x-1,y+1)][1])
        end
        for y in Ly:-1:2
            raw = mul(nn(CSE,(x,y+1)), TE[(x,y)])
            CSE[(x,y)] = raw*dag(PHS[(x-1,y)][1])
        end
    end
    # ---- N / S column strips: consume PHN / PHS ----
    for x in 1:Lx
        for y in 1:Ly-1
            raw = mul(nn(TN,(x,y)), A[(x,y)])
            x>=2      && (raw = raw*dag(PHN[(x-1,y+1)][1]))
            x<=Lx-1   && (raw = raw*PHN[(x,y+1)][1])
            TN[(x,y+1)]=raw
        end
        for y in Ly:-1:2
            raw = mul(A[(x,y)], nn(TS,(x,y+1)))
            x>=2      && (raw = raw*dag(PHS[(x-1,y)][1]))
            x<=Lx-1   && (raw = raw*PHS[(x,y)][1])
            TS[(x,y)]=raw
        end
    end
    return (;TW,TE,TN,TS,CNW,CNE,CSW,CSE,PHN,PHS,PVW,PVE)
end

function region_lnZ(E,A,Lx,Ly,cx,cy)
    rL=ceil(Int,cx); rR=floor(Int,cx)+1; tT=ceil(Int,cy); tB=floor(Int,cy)+1
    xint = (rL<rR); yint = (tT<tB)
    ts = Any[nn(E.CNW,(rL,tT)), nn(E.CNE,(rR,tT)), nn(E.CSW,(rL,tB)), nn(E.CSE,(rR,tB))]
    xint && push!(ts, nn(E.TN,(Int(cx),tT))); xint && push!(ts, nn(E.TS,(Int(cx),tB)))
    yint && push!(ts, nn(E.TW,(rL,Int(cy)))); yint && push!(ts, nn(E.TE,(rR,Int(cy))))
    (xint&&yint) && push!(ts, A[(Int(cx),Int(cy))])
    v = filter(!isnothing, ts)
    return log(real(scalar(reduce(*, v))))
end

function cvm_lnZ(E,A,Lx,Ly)
    F=0.0
    for x in 1:Lx,   y in 1:Ly;    F += region_lnZ(E,A,Lx,Ly,x,y); end
    for x in 1:Lx-1, y in 1:Ly;    F -= region_lnZ(E,A,Lx,Ly,x+0.5,y); end
    for x in 1:Lx,   y in 1:Ly-1;  F -= region_lnZ(E,A,Lx,Ly,x,y+0.5); end
    for x in 1:Lx-1, y in 1:Ly-1;  F += region_lnZ(E,A,Lx,Ly,x+0.5,y+0.5); end
    return F
end

# ===================== STATIONARY ITERATION =====================
# Projectors derived from the ENLARGED CORNER  C̃ = C · T_N · T_W · A  (corner + adjoining
# T's + vertex tensor), eigendecomposed on each of its two open interfaces. This needs the
# T's, which are themselves renormalized by those projectors -> circular -> iterate.
#   NW derives PHN (right iface) + PVW (down iface); SE derives PHS + PVE; NE/SW consume.
function sweep(A,hb,vb,Lx,Ly,χ,S)
    TW=Dict{Tuple{Int,Int},Any}(); TE=Dict{Tuple{Int,Int},Any}()
    TN=Dict{Tuple{Int,Int},Any}(); TS=Dict{Tuple{Int,Int},Any}()
    CNW=Dict{Tuple{Int,Int},Any}(); CNE=Dict{Tuple{Int,Int},Any}()
    CSW=Dict{Tuple{Int,Int},Any}(); CSE=Dict{Tuple{Int,Int},Any}()
    PHN=Dict{Tuple{Int,Int},Tuple{ITensor,Index}}(); PHS=Dict{Tuple{Int,Int},Tuple{ITensor,Index}}()
    PVW=Dict{Tuple{Int,Int},Tuple{ITensor,Index}}(); PVE=Dict{Tuple{Int,Int},Tuple{ITensor,Index}}()

    ow(d,k) = haskey(d,k) ? d[k][2] : nothing        # OLD interface index
    # ---- NW pass (all inputs from previous state S). Derives PHN, PVW ----
    for y in 1:Ly-1
        for x in 1:Lx-1
            Ct = mul(mul(mul(nn(S.CNW,(x,y)), nn(S.TN,(x,y))), nn(S.TW,(x,y))), A[(x,y)])
            ri=Index[]; (y>=2 && !isnothing(ow(S.PHN,(x,y)))) && push!(ri,ow(S.PHN,(x,y))); push!(ri,hb[(x,y)])
            Pr,wr=derive_proj(Ct,ri,χ); PHN[(x,y+1)]=(Pr,wr)
            di=Index[]; (x>=2 && !isnothing(ow(S.PVW,(x,y)))) && push!(di,ow(S.PVW,(x,y))); push!(di,vb[(x,y)])
            Pd,wd=derive_proj(Ct,di,χ); PVW[(x+1,y)]=(Pd,wd)
            CNW[(x+1,y+1)] = Ct*Pr*Pd
            raw = mul(nn(S.TW,(x,y)), A[(x,y)])
            y>1 && (raw = raw*PVW[(x+1,y-1)][1])
            TW[(x+1,y)] = raw*Pd
        end
    end
    for x in 1:Lx-1                                   # bottom row: TW has no down iface
        raw = mul(nn(S.TW,(x,Ly)), A[(x,Ly)])
        Ly>1 && (raw = raw*PVW[(x+1,Ly-1)][1])
        TW[(x+1,Ly)] = raw
    end
    # ---- SE pass. Derives PHS, PVE ----
    for y in Ly:-1:2
        for x in Lx:-1:2
            Ct = mul(mul(mul(nn(S.CSE,(x+1,y+1)), nn(S.TE,(x+1,y))), nn(S.TS,(x,y+1))), A[(x,y)])
            li=Index[hb[(x-1,y)]]; (y<=Ly-1 && !isnothing(ow(S.PHS,(x-1,y+1)))) && push!(li,ow(S.PHS,(x-1,y+1)))
            Pl,wl=derive_proj(Ct,li,χ); PHS[(x-1,y)]=(Pl,wl)
            ui=Index[vb[(x,y-1)]]; (x<=Lx-1 && !isnothing(ow(S.PVE,(x+1,y-1)))) && push!(ui,ow(S.PVE,(x+1,y-1)))
            Pu,wu=derive_proj(Ct,ui,χ); PVE[(x,y-1)]=(Pu,wu)
            CSE[(x,y)] = Ct*Pl*Pu
            raw = mul(A[(x,y)], nn(S.TE,(x+1,y)))
            y<Ly && (raw = raw*PVE[(x,y)][1])
            TE[(x,y)] = raw*Pu
        end
    end
    for x in Lx:-1:2                                  # top row: TE has no up iface
        raw = mul(A[(x,1)], nn(S.TE,(x+1,1)))
        Ly>1 && (raw = raw*PVE[(x,1)][1])
        TE[(x,1)] = raw
    end
    # ---- rebuild N / S column strips with the new PHN / PHS ----
    for x in 1:Lx
        for y in 1:Ly-1
            raw = mul(nn(S.TN,(x,y)), A[(x,y)])
            x>=2    && (raw = raw*dag(PHN[(x-1,y+1)][1]))
            x<=Lx-1 && (raw = raw*PHN[(x,y+1)][1])
            TN[(x,y+1)]=raw
        end
        for y in Ly:-1:2
            raw = mul(A[(x,y)], nn(S.TS,(x,y+1)))
            x>=2    && (raw = raw*dag(PHS[(x-1,y)][1]))
            x<=Lx-1 && (raw = raw*PHS[(x,y)][1])
            TS[(x,y)]=raw
        end
    end
    # ---- NE / SW: consume the derived families ----
    for y in 1:Ly-1
        for x in Lx:-1:2
            Ct = mul(mul(mul(nn(S.CNE,(x+1,y)), nn(S.TN,(x,y))), nn(S.TE,(x+1,y))), A[(x,y)])
            x>=2    && (Ct = Ct*dag(PHN[(x-1,y+1)][1]))
            y<=Ly-1 && (Ct = Ct*dag(PVE[(x,y)][1]))
            CNE[(x,y+1)] = Ct
        end
    end
    for y in Ly:-1:2
        for x in 1:Lx-1
            Ct = mul(mul(mul(nn(S.CSW,(x,y+1)), nn(S.TS,(x,y+1))), nn(S.TW,(x,y))), A[(x,y)])
            x<=Lx-1 && (Ct = Ct*dag(PHS[(x,y)][1]))
            y>=2    && (Ct = Ct*dag(PVW[(x+1,y-1)][1]))
            CSW[(x+1,y)] = Ct
        end
    end
    return (;TW,TE,TN,TS,CNW,CNE,CSW,CSE,PHN,PHS,PVW,PVE)
end

function main(; Lx = 4, Ly = 4, nsweeps = 6)
    for D in (2, 3)
        A, hb, vb = rand_net(Lx, Ly, D; seed = 7)
        zex = log(real(scalar(reduce(*, [A[(x, y)] for x in 1:Lx, y in 1:Ly]))))
        @printf("\nrandom positive %dx%d D=%d  ln Z_exact=%.8f\n", Lx, Ly, D, zex)
        for χ in (2, 3, 4, 6, 8)
            E = build_env(A, hb, vb, Lx, Ly, χ)            # greedy grown init
            f0 = abs(cvm_lnZ(E, A, Lx, Ly) - zex)
            errs = Float64[]; S = E
            for _ in 1:nsweeps                             # sweep to the fixed point
                S = sweep(A, hb, vb, Lx, Ly, χ, S)
                push!(errs, abs(cvm_lnZ(S, A, Lx, Ly) - zex))
            end
            @printf("  χ=%-2d greedy=%.3e  sweeps: %s\n", χ, f0,
                    join([@sprintf("%.2e", e) for e in errs], " "))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
