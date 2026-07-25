# PROTOTYPE — validated, NOT yet wired into the engine.
#
# The Möbius-weighted stationary ("partial Schur") interface projector, for ONE interface
# family (PH[:N,x,y]) on a 4x4. Everything here is checked; what remains is generalising the
# region bookkeeping to the other three families (PH[:S], PV[:W], PV[:E]) and plumbing it into
# `sweep_vertex_environments` as a nested fixed point (G depends on the projectors it sets).
#
# WHAT IT ESTABLISHES
#   1. Each interface appears in SIX regions. Z_R is linear in Pi = P_A P_B, so
#      Z_R = Tr[E_R^T Pi] and  G = dF/dPi = sum_R c_R E_R / Z_R.
#      Verified two ways: Tr[E_R^T Pi] == Z_R exactly for all six regions, and a
#      finite-difference check on the weighted log-sum (ratio -> 1.000000).
#   2. Stationarity <=> Pi commutes with G'  <=>  kept subspace is G'-invariant.
#   3. The exactly-stationary rank-k Pi comes from a partial Schur decomposition of G'
#      plus one Sylvester solve. Residual 6e-15 vs 0.44 for the current projector.
#
# THE FIDDLY PART is which two blocks CARRY the interface in each region — it is not always
# two corners. For PH[:N,x,y]:
#      region                 west carrier      east carrier
#      plaquette (x+.5,y-.5)  C_NW(x+1,y)       C_NE(x+1,y)
#      h-edge    (x+.5,y)     C_NW(x+1,y)       C_NE(x+1,y)
#      vertex    (x,y)        T_N(x,y)          C_NE(x+1,y)
#      vertex    (x+1,y)      C_NW(x+1,y)       T_N(x+1,y)
#      v-edge    (x,y-.5)     T_N(x,y)          C_NE(x+1,y)
#      v-edge    (x+1,y-.5)   C_NW(x+1,y)       T_N(x+1,y)
# Getting west/east backwards silently pairs P_B with P_B and the identity check fails.
#
# SIGN TRAP: Julia's `sylvester(A,B,C)` solves A*X + X*B + C = 0. The commutator
# [T, [[I,X],[0,0]]] vanishes iff T11*X - X*T22 == T12, so the call needs -T12:
#      X = sylvester(T11, -T22, -T12)
# With +T12 the "stationary" projector comes out with a residual 2x WORSE than the current one.
#
# Run: julia --project=. --startup-file=no examples/ctm_stationary_projector_prototype.jl

using TensorNetworkQuantumSimulator, ITensors, LinearAlgebra, Printf, Random
const T = TensorNetworkQuantumSimulator
Random.seed!(7)
L, D, χ = 4, 3, 6
tn = random_tensornetwork(Float64, named_grid((L,L)); bond_dimension=D)
conv = update(CTMEnvironmentCache(tn, χ); maxiter=25, tolerance=1e-12)
S, tbl = environments(conv), T._ctm_factor_table(conv)
x, y = 2, 3
KEY = (:N, x, y)

E(s,i,j) = T._ctm_enlarged(S, tbl, s, i, j)
nn(d,k)=get(d,k,nothing); A(i,j)=haskey(tbl,(i,j)) ? T._ctm_contract(tbl[(i,j)]) : nothing
mul(a,b)= isnothing(a) ? b : (isnothing(b) ? a : T._ctm_contract(ITensor[a,b]))
Bw, Be = E(:NW,x+1,y), E(:NE,x+1,y)
ins = collect(commoninds(Bw,Be)); insp = prime.(ins)
PR = T._ctm_interface_proj2(Bw, Be, ins, χ)
Pi = PR[1] * replaceinds(PR[2], ins, insp)

# `open` = leave KEY unprojected. Otherwise use the FRESH pair for KEY, S's for the rest.
ph(k, open) = (k == KEY ? (open ? nothing : PR) : nn(S.PH, k))
pv(k)       = nn(S.PV, k)
aA(t,p)= isnothing(p)||isnothing(t) ? t : t*p[1]
aB(t,p)= isnothing(p)||isnothing(t) ? t : t*p[2]
CNW(i,j,o)=aA(aA(E(:NW,i,j), ph((:N,i-1,j),o)), pv((:W,i,j-1)))
CNE(i,j,o)=aA(aB(E(:NE,i,j), ph((:N,i-1,j),o)), pv((:E,i,j-1)))
CSW(i,j,o)=aB(aA(E(:SW,i,j), ph((:S,i-1,j),o)), pv((:W,i,j-1)))
CSE(i,j,o)=aB(aB(E(:SE,i,j), ph((:S,i-1,j),o)), pv((:E,i,j-1)))
TN(i,j,o) =(r=mul(nn(S.T,(:N,i,j-1)), A(i,j-1)); aA(aB(r, ph((:N,i-1,j),o)), ph((:N,i,j),o)))
TS(i,j,o) =(r=mul(A(i,j), nn(S.T,(:S,i,j+1))); aA(aB(r, ph((:S,i-1,j),o)), ph((:S,i,j),o)))
TW(i,j)   =(r=mul(nn(S.T,(:W,i-1,j)), A(i-1,j)); aA(aB(r, pv((:W,i,j-1))), pv((:W,i,j))))
TE(i,j)   =(r=mul(A(i,j), nn(S.T,(:E,i+1,j))); aA(aB(r, pv((:E,i,j-1))), pv((:E,i,j))))

# each region: (weight, label, west-carrier, east-carrier, other blocks)
R = [(+1,"plaquette (x+½,y-½)", o->CNW(x+1,y,o), o->CNE(x+1,y,o),
        o->[CSW(x+1,y,o), CSE(x+1,y,o)]),
     (-1,"h-edge    (x+½,y)  ", o->CNW(x+1,y,o), o->CNE(x+1,y,o),
        o->[CSW(x+1,y+1,o), CSE(x+1,y+1,o), TW(x+1,y), TE(x+1,y)]),
     (+1,"vertex    (x,y)    ", o->TN(x,y,o),    o->CNE(x+1,y,o),
        o->[CNW(x,y,o), CSW(x,y+1,o), CSE(x+1,y+1,o), TS(x,y+1,o), TW(x,y), TE(x+1,y), A(x,y)]),
     (+1,"vertex    (x+1,y)  ", o->CNW(x+1,y,o), o->TN(x+1,y,o),
        o->[CNE(x+2,y,o), CSW(x+1,y+1,o), CSE(x+2,y+1,o), TS(x+1,y+1,o), TW(x+1,y), TE(x+2,y), A(x+1,y)]),
     (-1,"v-edge    (x,y-½)  ", o->TN(x,y,o),    o->CNE(x+1,y,o),
        o->[CNW(x,y,o), CSW(x,y,o), CSE(x+1,y,o), TS(x,y,o)]),
     (-1,"v-edge    (x+1,y-½)", o->CNW(x+1,y,o), o->TN(x+1,y,o),
        o->[CNE(x+2,y,o), CSW(x+1,y,o), CSE(x+2,y,o), TS(x+1,y,o)])]

println("region                 w    Z_R            Tr[E_R Pi] match")
Gs=ITensor[]; Zs=Float64[]; cs=Int[]
for (c,lbl,wf,ef,of) in R
    ZR = scalar(T._ctm_contract(ITensor[t for t in [wf(false),ef(false),of(false)...] if !isnothing(t)]))
    west = wf(true); east = replaceinds(ef(true), ins, insp)
    ER = T._ctm_contract(ITensor[t for t in [west,east,of(true)...] if !isnothing(t)])
    chk = scalar(ER*Pi)
    @printf("%s  %+d  %+.6e   %s\n", lbl, c, ZR,
        isapprox(chk,ZR;rtol=1e-9) ? "yes" : @sprintf("NO rel %.2e", abs(chk-ZR)/abs(ZR)))
    push!(Gs,ER); push!(Zs,ZR); push!(cs,c)
end
G = sum(cs[i]*Gs[i]/Zs[i] for i in eachindex(Gs))
Random.seed!(99); dP = random_itensor(inds(Pi)...)
F0 = sum(cs[i]*log(abs(Zs[i])) for i in eachindex(Gs))
println()
for ε in (1e-5,1e-6,1e-7)
    Fp = sum(cs[i]*log(abs(scalar(Gs[i]*(Pi+ε*dP)))) for i in eachindex(Gs))
    pred = ε*scalar(G*dP)
    @printf("ε=%.0e  ΔF=%+.6e  pred=%+.6e  ratio %.6f\n", ε, Fp-F0, pred, (Fp-F0)/pred)
end

# ---- exact stationarity residual, and the partial-Schur alternative ----------------
co = combiner(ins...); io = combinedind(co)
cop = combiner(insp...); iop = combinedind(cop)
tomat(t) = Array(t * co * cop, io, iop)
Gm, Pm = tomat(G), tomat(Pi)
n, k = size(Pm, 1), ITensors.dim(PR[3])
@printf("\ninterface PH[:N,%d,%d]:  n = %d,  k = %d\n", x, y, n, k)
@printf("Pi idempotent? ||Pi^2-Pi||/||Pi|| = %.2e   rank = %d\n",
        norm(Pm*Pm - Pm)/norm(Pm), rank(Pm; atol=1e-8*norm(Pm)))

Q = I(n) - Pm
res = (norm(Pm * Gm' * Q) + norm(Q * Gm' * Pm)) / norm(Gm)
@printf("STATIONARITY RESIDUAL  (||Pi G' Q|| + ||Q G' Pi||)/||G|| = %.4e\n", res)

# partial Schur: reorder so the k dominant eigenvalues of G' come first
Fs = schur(Matrix(Gm'))
λ = Fs.values; ord = sortperm(abs.(λ); rev=true)
sel = falses(n); sel[ord[1:k]] .= true
try
    Fo = ordschur(Fs, sel)
    Q1 = Fo.Z[:, 1:k]
    T11, T12, T22 = Fo.T[1:k,1:k], Fo.T[1:k,k+1:n], Fo.T[k+1:n,k+1:n]
    X = sylvester(T11, -T22, -T12)
    Pnew = Fo.Z * [I(k) X; zeros(n-k,k) zeros(n-k,n-k)] * Fo.Z'
    @printf("Schur Pi: idempotent %.2e   rank %d\n",
            norm(Pnew*Pnew-Pnew)/norm(Pnew), rank(Pnew; atol=1e-8*norm(Pnew)))
    resS = (norm(Pnew*Gm'*(I(n)-Pnew)) + norm((I(n)-Pnew)*Gm'*Pnew))/norm(Gm)
    @printf("Schur residual = %.4e   (%.1fx smaller)\n", resS, res/resS)
    # how different is the subspace? principal angles between column spaces
    sv = svd(Matrix(qr(Q1).Q)[:,1:k]' * Matrix(qr(Pm).Q)[:,1:k]).S
    @printf("principal angles (deg): %s\n",
            join([@sprintf("%.1f", acosd(clamp(s,-1,1))) for s in sv], " "))
catch e
    println("Schur step failed: ", sprint(showerror, e)[1:min(end,120)])
end
