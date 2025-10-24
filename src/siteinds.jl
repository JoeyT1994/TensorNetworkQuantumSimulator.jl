using Dictionaries: Dictionary

function siteinds(g::AbstractGraph, sitetype::String, sitedimension::Int = site_dimension(sitetype))
    vs = collect(vertices(g))
    return Dictionary(vs, [Index[Index(sitedimension, site_tag(sitetype))] for v in vs])
end

function site_dimension(sitetype::String)
    sitetype = replace(lowercase(sitetype), " " => "")
    sitetype ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"] && return 2
    sitetype ∈ ["qutrit", "s=1", "spin1"]  && return 3
    sitetype ∈ ["pauli"] && return 4
    error("Don't know what physical space that site type should be")
end

function site_tag(sitetype::String)
    sitetype = replace(lowercase(sitetype), " " => "")
    sitetype ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"] && return "S=1/2"
    sitetype ∈ ["qutrit", "s=1", "spin1"] && return "S=1"
    sitetype ∈ ["pauli"] && return "Pauli"
    error("Don't know how to interpret that site type. Supported: S=1/2, S=1, Pauli")
end
