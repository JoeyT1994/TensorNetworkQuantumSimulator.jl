
function default_siteinds(g::AbstractGraph; kwargs...)
    return siteinds("S=1/2", g; kwargs...)
end

"""
    siteinds(sitetype::String, g::AbstractGraph; inds_per_site = 1, symmetry = nothing, sectors = nothing)

Site indices for every vertex of `g`. `sitetype` is one of `"S=1/2"`, `"S=1"`,
`"Fermion"`, `"SpinfulFermion"` (aliases: `"Electron"`).

Pass `symmetry` for structurally-enforced abelian conservation: `"Z2"`, `"U1"` for
spins; `"fZ2"` (parity), `"fU1"` (particle number), `"fU1xU1"` (separate N↑, N↓) for
fermions. Fermionic site types are graded by default (`"fZ2"` when no `symmetry` is
given). Named site types derive their sector decomposition automatically; pass
`sectors` (charge => dimension pairs summing to the site dimension) only for custom
sites. With an even `inds_per_site` (purifications) the second half of each vertex's
indices are dual-representation ancilla copies.
"""
function siteinds(sitetype::String, g::AbstractGraph, sitedimension::Integer = site_dimension(sitetype); inds_per_site::Integer = 1, sectors = nothing, symmetry::Union{String, Nothing} = nothing)
    vs = collect(vertices(g))
    st = replace(lowercase(sitetype), " " => "")
    fermionic = st ∈ ["fermion", "spinlessfermion", "spinfulfermion", "electron"]
    #fermionic sites are graded by default (parity at minimum); a `symmetry` alone
    #suffices for named site types — the physical basis fixes the sector decomposition
    fermionic && symmetry === nothing && (symmetry = "fZ2")
    sectors === nothing && symmetry !== nothing &&
        (sectors = default_sectors(sitetype, symmetry))
    sectors !== nothing &&
        return graded_siteinds(sitetype, vs, sitedimension, sectors, symmetry, inds_per_site)
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
