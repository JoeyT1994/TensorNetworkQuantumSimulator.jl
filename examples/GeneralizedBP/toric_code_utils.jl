using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Dictionaries
using ITensors
using NamedGraphs

function toric_code_ground_state(n::Integer)
    g = named_grid((n, n); periodic = true)
    vs = collect(vertices(g))
    tensors = Dictionary{vertextype(g), ITensor}()
    s = siteinds("S=1/2", g)
    es=  edges(g)
    e_dict = Dictionary(es, [Index(2) for e in edges(g)])
    e_dict = merge(e_dict, Dictionary(reverse.(es), collect(values(e_dict))))

    for v in vertices(g)
        incoming_es = filter(e -> v == src(e) || v == dst(e), es)
        incoming_inds = [e_dict[e] for e in incoming_es]
        sv = only(s[v])

        state = ITensor(ComplexF64, 0.0, [incoming_inds... , sv])

        north_index = e_dict[NamedEdge((mod1(v[1]+1, n), v[2]) => v)]
        east_index = e_dict[NamedEdge((v[1], mod1(v[2]+1, n)) => v)]
        south_index = e_dict[NamedEdge(v => (mod1(v[1]-1, n), v[2]))]
        west_index = e_dict[NamedEdge(v => (v[1], mod1(v[2]-1, n)))]

        if iseven(sum(v))
            state  = state + (ITensors.onehot(north_index => 1) * ITensors.onehot(east_index => 1) + ITensors.onehot(north_index => 2) * ITensors.onehot(east_index => 2)) * (ITensors.onehot(south_index => 1) * ITensors.onehot(west_index => 1) + ITensors.onehot(south_index => 2) * ITensors.onehot(west_index => 2)) * ITensors.onehot(sv => 1)

            state  = state + (ITensors.onehot(north_index => 1) * ITensors.onehot(east_index => 1) - ITensors.onehot(north_index => 2) * ITensors.onehot(east_index => 2)) * (ITensors.onehot(south_index => 1) * ITensors.onehot(west_index => 1) - ITensors.onehot(south_index => 2) * ITensors.onehot(west_index => 2)) * ITensors.onehot(sv => 2)
        else
            state  = state + (ITensors.onehot(north_index => 1) * ITensors.onehot(west_index => 1) + ITensors.onehot(north_index => 2) * ITensors.onehot(west_index => 2)) * (ITensors.onehot(south_index => 1) * ITensors.onehot(east_index => 1) + ITensors.onehot(south_index => 2) * ITensors.onehot(east_index => 2)) * ITensors.onehot(sv => 1)

            state  = state + (ITensors.onehot(north_index => 1) * ITensors.onehot(west_index => 1) - ITensors.onehot(north_index => 2) * ITensors.onehot(west_index => 2)) * (ITensors.onehot(south_index => 1) * ITensors.onehot(east_index => 1) - ITensors.onehot(south_index => 2) * ITensors.onehot(east_index => 2)) * ITensors.onehot(sv => 2)
        end
        set!(tensors, v, state)
    end

    return TensorNetworkState(TensorNetwork(tensors, g), s)
end

function get_x_spider()
    myinds = [Index(2) for j=1:4]
    x_spider = ITensor(Int64, myinds...)
    for idx=CartesianIndices((2,2,2,2))
        if sum(Tuple(idx) .- 1)%2==0
	    x_spider[[myinds[j]=>idx[j] for j=1:4]...] = 1
	end
    end

    return x_spider
end
function get_flat_vertex()
    myinds = [Index(2) for j=1:8]
    x_spider = ITensor(Int64, myinds[1:4]...)
    for idx=CartesianIndices((2,2,2,2))
        if sum(Tuple(idx) .- 1)%2==0
	    x_spider[[myinds[j]=>idx[j] for j=1:4]...] = 1
	end
    end
    site_ten = delta(myinds[1], myinds[1]', myinds[5]) * delta(myinds[2],myinds[2]', myinds[6]) * delta(myinds[3], myinds[3]',myinds[7]) * delta(myinds[4], myinds[4]',myinds[8]) * x_spider * x_spider'
    return site_ten
end

function toric_code_flat(n::Integer)
    g = named_grid((n,n); periodic = true)
    vs = collect(vertices(g))

    x_spider = get_x_spider()
    
    l = Dict(e => Index(2) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        is = [l[NamedEdge(v => vn)] for vn in neighbors(g, v)]
        set!(tensors, v, replaceinds(x_spider, inds(x_spider), is))
    end
    return TensorNetwork(tensors, g)
end