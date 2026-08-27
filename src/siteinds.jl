using Dictionaries: Dictionary

function default_siteinds(g::AbstractGraph; kwargs...)
    return siteinds("S=1/2", g; kwargs...)
end

function siteinds(sitetype::String, g::AbstractGraph, sitedimension::Integer = site_dimension(sitetype); inds_per_site::Integer = 1, sectors = nothing, symmetry::String = "U1")
    vs = collect(vertices(g))
    if sectors !== nothing
        #graded (symmetric, TensorKit-backed) site indices: `sectors` is a list of
        #charge => dimension pairs under the group named by `symmetry` ("Z2"/"U1"/"fZ2")
        sum(last.(sectors)) == sitedimension ||
            error("siteinds: sector dimensions $(sectors) do not sum to the site dimension $(sitedimension)")
        sp = KTensors.graded_space(symmetry, sectors)
        return Dictionary(vs, [[KTensors.KIndex(sp, site_tag(sitetype)) for i in 1:inds_per_site] for v in vs])
    end
    if replace(lowercase(sitetype), " " => "") ∈ ["fermion", "spinlessfermion"]
        #fermionic (Z2-parity, TensorKit-backed) site indices: |0⟩ even, |1⟩ odd
        return Dictionary(vs, [[KTensors.new_fermion_index(1, 1; tags = site_tag(sitetype)) for i in 1:inds_per_site] for v in vs])
    end
    return Dictionary(vs, [[new_index(sitedimension; tags = site_tag(sitetype)) for i in 1:inds_per_site] for v in vs])
end

function site_dimension(sitetype::String)
    sitetype = replace(lowercase(sitetype), " " => "")
    sitetype ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"] && return 2
    sitetype ∈ ["fermion", "spinlessfermion"] && return 2
    sitetype ∈ ["qutrit", "s=1", "spin1"]  && return 3
    error("Don't know what physical space that site type should be")
end

function site_tag(sitetype::String)
    sitetype = replace(lowercase(sitetype), " " => "")
    sitetype ∈ ["s=1/2", "qubit", "spin1/2", "spinhalf"] && return "S=1/2"
    sitetype ∈ ["qutrit", "s=1", "spin1"] && return "S=1"
    sitetype ∈ ["fermion", "spinlessfermion"] && return "Fermion"
    error("Don't know how to interpret that site type. Supported: S=1/2, S=1, Fermion.")
end
