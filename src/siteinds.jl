using Dictionaries: Dictionary

function default_siteinds(g::AbstractGraph; kwargs...)
    return siteinds("S=1/2", g; kwargs...)
end

function siteinds(sitetype::String, g::AbstractGraph, sitedimension::Integer = site_dimension(sitetype); inds_per_site::Integer = 1, sectors = nothing, symmetry::Union{String, Nothing} = nothing)
    vs = collect(vertices(g))
    st = replace(lowercase(sitetype), " " => "")
    fermionic = st ∈ ["fermion", "spinlessfermion", "spinfulfermion", "electron"]
    #fermionic sites are graded by default (parity at minimum); a `symmetry` alone
    #suffices for named site types — the physical basis fixes the sector decomposition
    fermionic && symmetry === nothing && (symmetry = "fZ2")
    sectors === nothing && symmetry !== nothing &&
        (sectors = default_sectors(sitetype, symmetry))
    if sectors !== nothing
        #graded (symmetric, TensorKit-backed) site indices: `sectors` is a list of
        #charge => dimension pairs under the group named by `symmetry`
        symmetry === nothing && error("siteinds: explicit `sectors` need a `symmetry` name")
        sum(last.(sectors)) == sitedimension ||
            error("siteinds: sector dimensions $(sectors) do not sum to the site dimension $(sitedimension)")
        sp = Tensors.graded_space(symmetry, sectors)
        #with an even number of inds per site (purifications), the second half are
        #ancillas and carry the DUAL representation (dag'd copies) so the identity
        #state is flux-zero per site
        anc(i) = iseven(inds_per_site) && i > inds_per_site ÷ 2
        return Dictionary(vs, [[(ind = Tensors.Index(sp, site_tag(sitetype)); anc(i) ? dag(ind) : ind) for i in 1:inds_per_site] for v in vs])
    end
    return Dictionary(vs, [[new_index(sitedimension; tags = site_tag(sitetype)) for i in 1:inds_per_site] for v in vs])
end

#The sector decomposition of a named site type's physical basis under a symmetry —
#users only need `sectors` for custom sites.
function default_sectors(sitetype::String, symmetry::String)
    st = replace(lowercase(sitetype), " " => "")
    sym = replace(lowercase(symmetry), " " => "")
    if st ∈ ["fermion", "spinlessfermion"]
        #|0⟩, |1⟩: charge = occupation (parity or particle number)
        sym ∈ ["fz2", "fermion", "fermionparity", "fu1", "fermionnumber"] && return [0 => 1, 1 => 1]
    elseif st ∈ ["spinfulfermion", "electron"]
        #|0⟩, |↑⟩, |↓⟩, |↑↓⟩
        sym ∈ ["fz2", "fermion", "fermionparity"] && return [0 => 2, 1 => 2]
        sym ∈ ["fu1", "fermionnumber"] && return [0 => 1, 1 => 2, 2 => 1]
        sym ∈ ["fu1xu1", "fu1u1"] && return [(0, 0) => 1, (1, 0) => 1, (0, 1) => 1, (1, 1) => 1]
    elseif st ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"]
        #|↑⟩, |↓⟩: Z2 charge = spin flip parity; U1 charge = 2Sz
        sym == "z2" && return [0 => 1, 1 => 1]
        sym ∈ ["u1", "u(1)"] && return [1 => 1, -1 => 1]
    end
    return error(
        "siteinds: no default sector decomposition for site type \"$sitetype\" under " *
            "symmetry \"$symmetry\" — pass `sectors` (charge => dimension pairs) explicitly"
    )
end

function site_dimension(sitetype::String)
    sitetype = replace(lowercase(sitetype), " " => "")
    sitetype ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"] && return 2
    sitetype ∈ ["fermion", "spinlessfermion"] && return 2
    sitetype ∈ ["spinfulfermion", "electron"] && return 4
    sitetype ∈ ["qutrit", "s=1", "spin1"]  && return 3
    error("Don't know what physical space that site type should be")
end

function site_tag(sitetype::String)
    sitetype = replace(lowercase(sitetype), " " => "")
    sitetype ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"] && return "S=1/2"
    sitetype ∈ ["qutrit", "s=1", "spin1"] && return "S=1"
    sitetype ∈ ["fermion", "spinlessfermion"] && return "Fermion"
    sitetype ∈ ["spinfulfermion", "electron"] && return "SpinfulFermion"
    error("Don't know how to interpret that site type. Supported: S=1/2, S=1, Fermion, SpinfulFermion.")
end
