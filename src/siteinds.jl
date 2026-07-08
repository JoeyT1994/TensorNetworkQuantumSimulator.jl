using Dictionaries: Dictionary

function default_siteinds(g::AbstractGraph; kwargs...)
    return siteinds("S=1/2", g; kwargs...)
end

function siteinds(sitetype::String, g::AbstractGraph, sitedimension::Integer = site_dimension(sitetype); inds_per_site::Integer = 1)
    vs = collect(vertices(g))
    return Dictionary{vertextype(g), Vector{<:Index}}(vs, [Index[ITensors.settags(Index(sitedimension), site_tag(sitetype)) for i in 1:inds_per_site] for v in vs])
end

function site_dimension(sitetype::String)
    sitetype = replace(lowercase(sitetype), " " => "")
    sitetype ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"] && return 2
    sitetype ∈ ["qutrit", "s=1", "spin1"]  && return 3
    error("Don't know what physical space that site type should be")
end

function site_tag(sitetype::String)
    sitetype = replace(lowercase(sitetype), " " => "")
    sitetype ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"] && return "S=1/2"
    sitetype ∈ ["qutrit", "s=1", "spin1"] && return "S=1"
    error("Don't know how to interpret that site type. Supported: S=1/2, S=1.")
end
