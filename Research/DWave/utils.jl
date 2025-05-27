using TensorNetworkQuantumSimulator: BoundaryMPSCache
using ITensorNetworks: IndsNetwork, BeliefPropagationCache
using NamedGraphs
using NamedGraphs: AbstractGraph, NamedGraph
using NamedGraphs.GraphsExtensions: add_edge
using Graphs

const G = Graphs
const NG = NamedGraphs

using CSV
using Interpolations
using Tables

global gammas, Js =  CSV.File(joinpath(@__DIR__, "AnnealingSchedules/Gammas.csv"); select=[1, 2], delim=",", header=0), CSV.File(joinpath(@__DIR__, "AnnealingSchedules/Js.csv"); select=[1, 2], delim=",", header=0)

global gamma_xvals, gamma_yvals = Tables.getcolumn(gammas,1), Tables.getcolumn(gammas,2)
global Js_xvals, Js_yvals = Tables.getcolumn(Js,1), Tables.getcolumn(Js,2)


global gamma_interpolation, J_interpolation = linear_interpolation(gamma_xvals, gamma_yvals, extrapolation_bc=Line()), linear_interpolation(Js_xvals, Js_yvals, extrapolation_bc=Line())

function annealing_schedule(current_annealing_time, maximum_annealing_time)
    x = current_annealing_time / maximum_annealing_time
    Γ, J = gamma_interpolation(x), J_interpolation(x)
    Γ < 0 && return 0.0, J
    return Γ, J
end


function zz_dict_boundarymps(ψ::ITensorNetwork, ψIψ_boundarymps::BoundaryMPSCache; vertical_only = true)
    out = Dictionary()
    for e in edges(ψ)
        if !vertical_only || (first(src(e)) == first(dst(e)))
            set!(out, e, expect(ψIψ_boundarymps, "Z", src(e), dst(e)))
        end
    end
    return out
end

function column_aligned_zz_dict_boundarymps(ψ::ITensorNetwork, ψIψ_boundarymps::BoundaryMPSCache)
    cols = ITensorNetworks.partitions(ψIψ_boundarymps)
    s = siteinds(ψ)
    out = Dictionary()
    for c in cols
        szszs_c = column_aligned_zzs(s, ψIψ_boundarymps, c)
        out = merge(out, szszs_c)
    end
    return out
end

function column_aligned_zzs(s::IndsNetwork, ψIψ::BoundaryMPSCache, col)
    vs = ITensorNetworks.planargraph_vertices(ψIψ, col)
    ψIψ = partition_update(ψIψ, col)
    szszs = Dictionary()
    for (i,v1) in enumerate(vs)
        v_cur = v1
        pg = copy(partitioned_tensornetwork(ψIψ))
        pg = rem_vertex(pg, (v1, "operator"))
        ψIψ_deleted = BoundaryMPSCache(BeliefPropagationCache(pg, messages(ψIψ)), ITensorNetworks.ppg(ψIψ), ITensorNetworks.maximum_virtual_dimension(ψIψ))
        for (j, v2) in enumerate(vs[i+1:length(vs)])
          ψIψ_deleted = partition_update(ψIψ_deleted, [v_cur], [v2])
          ρ = contract(environment(bp_cache(ψIψ_deleted), [(v2, "operator")]); sequence = "automatic")
          v_cur = v2

          ρ = permute(ρ, reduce(vcat, [s[v1], s[v2], s[v1]', s[v2]']))
          szsz = (ρ * ITensors.op("Z", s[v1]) * ITensors.op("Z", s[v2]))[] / (ρ * ITensors.op("I", s[v1]) * ITensors.op("I", s[v2]))[]
          set!(szszs, NamedEdge(v1 => v2), szsz)
        end
    end
    return szszs
end


function DWave_vertex_mapping(g::NamedGraph, radius::Int)
    forward_dict = Dictionary()
    backward_dict = Dictionary()
    for v in vertices(g)
        set!(forward_dict, v, (last(v)-1)*(radius) + (first(v)-1))
        set!(backward_dict, (last(v)-1)*(radius) + (first(v)-1), v)
    end
    return forward_dict, backward_dict
end

function couplings_to_edge_dict(g::NamedGraph, radius::Int, instance)
    couplings, is, js = instance["Jij"], instance["i"], instance["j"]
    _, backward_dict = DWave_vertex_mapping(g, radius)
    edge_coupling_dict = Dictionary()
    for n in 1:length(couplings)
        Jij, i, j = couplings[n], is[n], js[n]
        vsrc, vdst = backward_dict[i], backward_dict[j]
        set!(edge_coupling_dict, NamedEdge(vsrc => vdst), Jij)
    end
    return edge_coupling_dict
end

function convert_correlations_to_dict(g::NamedGraph, radius::Int, correlations)
    forward_dict, backward_dict = DWave_vertex_mapping(g, radius)
    correlation_dict = Dictionary()
    L = nv(g)
    n = 1
    for i in 0:(L-1)
        for j in (i+1):(L-1)
            vsrc, vdst = backward_dict[i], backward_dict[j]
            set!(correlation_dict, NamedEdge(vsrc => vdst), correlations[n])
            n += 1
        end
    end
    return correlation_dict
end

function convert_dict_to_correlations(g::NamedGraph, radius::Int, dict)
    forward_dict, backward_dict = DWave_vertex_mapping(g, radius)
    correlations = ComplexF64[]
    L = nv(g)
    n = 1
    for i in 0:(L-1)
        for j in (i+1):(L-1)
            vsrc, vdst = backward_dict[i], backward_dict[j]
            e =NamedEdge(vsrc => vdst)
            corr = e ∈ keys(dict) ? dict[e] : dict[reverse(e)]
            push!(correlations, corr)
        end
    end
    return correlations
end

function convert_dict_to_correlations(g, dict)
    L = nv(g)
    correlations = ComplexF64[]
    es = []
    for i in 0:(L-1)
        for j in (i+1):(L-1)
            e =NamedEdge(i + 1 => j + 1)
            corr = e ∈ keys(dict) ? dict[e] : dict[reverse(e)]
            push!(correlations, corr)
            push!(es, e)
        end
    end
    return correlations, es
end

function convert_dict_to_correlations(L::Int64, dict)
    correlations = ComplexF64[]
    es = []
    for i in 0:(L-1)
        for j in (i+1):(L-1)
            e =NamedEdge(i + 1 => j + 1)
            corr = e ∈ keys(dict) ? dict[e] : dict[reverse(e)]
            push!(correlations, corr)
            push!(es, e)
        end
    end
    return correlations, es
end

function insert_operators(ψIψ_boundarymps::BoundaryMPSCache, vs, strings)
    allequal(vs) && return ψIψ_boundarymps
    ψIψ_boundarymps = copy(ψIψ_boundarymps)
    for (v, string) in zip(vs, strings)
        current_op = only(factors(ψIψ_boundarymps, [(v, "operator")]))
        oper = ITensors.op(string, only(filter(i -> plev(i) == 0, inds(current_op))))
        ψIψ_boundarymps = update_factor(ψIψ_boundarymps, (v, "operator"), oper)
    end
    return ψIψ_boundarymps
end

function terms_to_scalar(numerator_numerator_terms, numerator_denominator_terms, denominator_numerator_terms, denominator_denominator_terms)
    return exp(sum(log.(numerator_numerator_terms)) - sum(log.(numerator_denominator_terms)) - sum(log.(denominator_numerator_terms)) + sum(log.(denominator_denominator_terms)))
end

#Assume symmetry
function zz_correlations(radius::Int64, ψ::ITensorNetwork, v; fit_kwargs = (; maxiter =5, message_update_kwargs = (; niters = 30, tolerance = 1e-10, verbosity = true)), mps_rank::Int64 = 1)
    ψIψ = BoundaryMPSCache(BeliefPropagationCache(QuadraticFormNetwork(ψ)); message_rank = mps_rank)
    s = only(inds(only(factors(ψIψ, [(v, "operator")])); plev = 0))
    ψIψ = update_factor(ψIψ, (v, "operator"), onehot(s => 1) * onehot(s' => 1))
    println("Updating")
    ψIψ = update(ψIψ; fit_kwargs...)
    println("Updated")
    corrs = Dictionary()
    for col in first(v):radius
        v_prev = []
        for vp in [[(col, row)] for row in 1:radius]
            ψIψ = isempty(v_prev) ? partition_update(ψIψ, vp) : partition_update(ψIψ, v_prev, vp)
            if first(vp) != v
                ρ = contract(environment(bp_cache(ψIψ), [(only(vp), "operator")]); sequence = "automatic")
                p_upup, p_downup = diag(ρ)[1], diag(ρ)[2]
                szsz = (p_upup - p_downup) / (p_upup + p_downup)
                set!(corrs, NamedEdge(v => first(vp)), szsz)
            end
            v_prev = vp
        end
    end
    return corrs
end

#Assume symmetry
function magnetisations(radius::Int64, ψ::ITensorNetwork; fit_kwargs = (; maxiter =5, message_update_kwargs = (; niters = 30, tolerance = 1e-10, verbosity = true)), mps_rank::Int64 = 1)
    ψIψ = BoundaryMPSCache(BeliefPropagationCache(QuadraticFormNetwork(ψ)); message_rank = mps_rank)
    println("Updating")
    ψIψ = update(ψIψ; fit_kwargs...)
    println("Updated")
    mags = Dictionary()
    for col in 1:radius
        v_prev = []
        for v in [[(col, row)] for row in 1:radius]
            ψIψ = isempty(v_prev) ? partition_update(ψIψ, v) : partition_update(ψIψ, v_prev, v)
            ρ = contract(environment(bp_cache(ψIψ), [(only(v), "operator")]); sequence = "automatic")
            p_up, p_down = diag(ρ)[1], diag(ρ)[2]
            sz = (p_up - p_down) / (p_up + p_down)
            set!(mags, only(v), sz)
        end
    end
    return mags
end

#Assume symmetry
function x_magnetisations(radius::Int64, ψIψ_bmps::BoundaryMPSCache)
    mags = Dictionary()
    ψIψ_bmps_t = copy(ψIψ_bmps)
    for col in 1:radius
        v_prev = []
        for v in [[(col, row)] for row in 1:radius]
            ψIψ_bmps_t = isempty(v_prev) ? partition_update(ψIψ_bmps_t, v) : partition_update(ψIψ_bmps_t, v_prev, v)
            ρ = contract(environment(bp_cache(ψIψ_bmps_t), [(only(v), "operator")]); sequence = "automatic")
            s_ind = only(filter(i -> plev(i) == 0, inds(ρ)))
            oz = ITensors.op("X", s_ind)
            ρ /= tr(ρ)
            sx = (ρ * oz)[]
            set!(mags, only(v), sx)
        end
    end
    return mags
end

#Assume symmetry
function zz_correlations_kz(radius::Int64, ψ::ITensorNetwork, v; fit_kwargs = (; maxiter =5, message_update_kwargs = (; niters = 30, tolerance = 1e-10)), mps_rank::Int64 = 1)
    ψIψ = BoundaryMPSCache(BeliefPropagationCache(QuadraticFormNetwork(ψ)); message_rank = mps_rank)
    s = only(inds(only(factors(ψIψ, [(v, "operator")])); plev = 0))
    pg = partitioned_tensornetwork(ψIψ)
    pg = rem_vertex(pg, (v, "operator"))
    ψIψ = BoundaryMPSCache(BeliefPropagationCache(pg, messages(ψIψ)); message_rank = mps_rank)
    ψIψ = update_factor(ψIψ, (v, "ket"), ψ[v]*onehot(s => 1))
    ψIψ = update_factor(ψIψ, (v, "bra"), dag(prime(ψ[v]))*onehot(s' => 1))
    println("Updating")
    ψIψ = update(ψIψ; fit_kwargs...)
    println("Updated")
    corrs = Dictionary()
    for col in 1:radius
        v_prev = []
        for vp in [[(col, row)] for row in 1:radius]
            ψIψ = isempty(v_prev) ? partition_update(ψIψ, vp) : partition_update(ψIψ, v_prev, vp)
            if first(vp) != v
                ρ = contract(environment(bp_cache(ψIψ), [(only(vp), "operator")]); sequence = "automatic")
                p_upup, p_downup = diag(ρ)[1], diag(ρ)[2]
                szsz = (p_upup - p_downup) / (p_upup + p_downup)
                set!(corrs, NamedEdge(v => first(vp)), szsz)
            end
            v_prev = vp
        end
    end
    return corrs
end

#Project spins on sites v1 and v2 to v1_val (1 = up, 2 = down) and v2_val
function project(ψ::ITensorNetwork, ψIψ::BeliefPropagationCache, v1, v2, v1_val::Int64 = 1, v2_val::Int64=1)
    ψIψ, ψ = copy(ψIψ), copy(ψ)
    s1 = only(inds(only(ITN.factors(ψIψ, [(v1, "operator")])); plev = 0))
    s2 = only(inds(only(ITN.factors(ψIψ, [(v2, "operator")])); plev = 0))
    ψIψ = ITN.update_factor(ψIψ, (v1, "operator"), onehot(s1 => v1_val) * dag(onehot(s1' => v1_val)))
    ψIψ = ITN.update_factor(ψIψ, (v2, "operator"), onehot(s2 => v2_val) * dag(onehot(s2' => v2_val)))

    return ψ, ψIψ
end 

#Log scalar contraction of bpc
function logscalar(bpc::BeliefPropagationCache)
    nums, denoms = ITN.scalar_factors_quotient(bpc)
    return sum(log.(nums)) - sum(log.(denoms))
end

function cumulative_weights(bpc::BeliefPropagationCache, egs::Vector{<:AbstractNamedGraph})
    isempty(egs) && [1]
    circuit_lengths = sort(unique(length.(edges.(egs))))
    outs = []
    for cl in circuit_lengths
        egs_cl = filter(eg -> length(edges(eg)) == cl, egs)
        sum_ws = sum(TN.weights(bpc, egs_cl))
        push!(outs, sum_ws)
    end
    outs = vcat([1.0], outs)
    return cumsum(outs)
end

#Compute zz with loop correction and all bells and whistles
function zz_correlation_bp_loopcorrectfull(ψ::ITensorNetwork, v1, v2, egs::Vector{<:AbstractNamedGraph})
    ψIψ = BeliefPropagationCache(ITN.QuadraticFormNetwork(ψ))

    ψ_upup, ψIψ_upup = project(ψ, ψIψ, v1, v2, 2, 2)
    ψIψ_upup = updatecache(ψIψ_upup)
    p_upup_bp = exp(logscalar(ψIψ_upup))
    _, ψIψ_upup  = normalize(ψ_upup, ψIψ_upup; update_cache = false)
    cfes = cumulative_weights(ψIψ_upup, egs)
    p_upups = [p_upup_bp*cfe for cfe in cfes]

    ψ_updown, ψIψ_updown = project(ψ, ψIψ, v1, v2, 2, 1)
    ψIψ_updown = updatecache(ψIψ_updown)
    p_updown_bp = exp(logscalar(ψIψ_updown))
    _, ψIψ_updown  = normalize(ψ_updown, ψIψ_updown; update_cache = false)
    cfes = cumulative_weights(ψIψ_updown, egs)
    p_updowns = [p_updown_bp*cfe for cfe in cfes]

    szszs = [(p_upup - p_updown) / (p_upup + p_updown) for (p_upup, p_updown) in zip(p_upups, p_updowns)]

    return szszs
end

function _quadraticformnetwork(ψ)
    s = siteinds(ψ)
    operator_inds = ITN.union_all_inds(s, prime(s))
    I = ITensorNetwork(collect(vertices(ψ)), [!isempty(operator_inds[v]) ? prod([delta(s_ind, prime(s_ind)) for s_ind in s[v]]) : ITensor(1.0) for v in vertices(ψ)])
    return ITN.QuadraticFormNetwork(I, ψ)
end

function zz_correlation_bp_loopcorrectfull_dimerized(old_sinds::IndsNetwork, ψ::ITensorNetwork, v1, v2, egs::Vector)
    vs = collect(vertices(ψ))
    v1_dimer, v2_dimer = only(filter(v -> last(v) == v1 || first(v) == v1, vs)), only(filter(v -> last(v) == v2 || first(v) == v2, vs))


    ψ_upup = copy(ψ)
    ψ_upup[v1_dimer] = ψ_upup[v1_dimer] * onehot(only(old_sinds[v1]) => 1)
    ψ_upup[v2_dimer] = ψ_upup[v2_dimer] * onehot(only(old_sinds[v2]) => 1)
    ψIψ_upup = _quadraticformnetwork(ψ_upup)
    ψIψ_upup = updatecache(BeliefPropagationCache(ψIψ_upup))
    p_upup_bp = exp(logscalar(ψIψ_upup))
    _, ψIψ_upup  = normalize(ψ_upup, ψIψ_upup; update_cache = false)
    cfes = cumulative_weights(ψIψ_upup, egs)
    p_upups = [p_upup_bp*cfe for cfe in cfes]

    ψ_updown = copy(ψ)
    ψ_updown[v1_dimer] = ψ_updown[v1_dimer] * onehot(only(old_sinds[v1]) => 1)
    ψ_updown[v2_dimer] = ψ_updown[v2_dimer] * onehot(only(old_sinds[v2]) => 2)
    ψIψ_updown = _quadraticformnetwork(ψ_updown)
    ψIψ_updown = updatecache(BeliefPropagationCache(ψIψ_updown))
    p_updown_bp = exp(logscalar(ψIψ_updown))
    _, ψIψ_updown  = normalize(ψ_updown, ψIψ_updown; update_cache = false)
    cfes = cumulative_weights(ψIψ_updown, egs)
    p_updowns = [p_updown_bp*cfe for cfe in cfes]
    
    szszs = [(p_upup - p_updown) / (p_upup + p_updown) for (p_upup, p_updown) in zip(p_upups, p_updowns)]

    return szszs
end

function graph_couplings_from_instance(file_name)
    d = npzread(file_name)
    is = d["i"]
    js = d["j"]
    edges = [NamedEdge(is[k] + 1 => js[k] + 1) for k in 1:length(is)]
    edge_couplings = Dictionary(edges, d["Jij"])
    verts = unique(vcat(src.(edges), dst.(edges)))
    g = NamedGraph(verts)
    g = NG.GraphsExtensions.add_edges(g, edges)
    return g, edge_couplings
end

function named_cylinder(nx::Int64, ny::Int64)
    g = named_grid((nx, ny))
    for i in 1:ny
        g = NG.GraphsExtensions.add_edge(g, NamedEdge((1, i) => (nx, i)))
    end
    return g
end