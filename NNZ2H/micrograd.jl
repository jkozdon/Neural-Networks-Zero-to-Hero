### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 2fc9f05e-b170-4575-8dd4-ac77f64494fc
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="GraphViz",
			url="https://github.com/abelsiqueira/GraphViz.jl",
		rev="38-add-engine-attribute"),
    ])
    using GraphViz
end


# ╔═╡ e5e1afe4-43ce-435e-b734-e7321ca8bfd8
mutable struct Value{T <: AbstractFloat}
	data::T
	grad::T
	label::Union{String, Nothing}
	children::Union{
		Tuple{Symbol,
			Value{T},
			Union{Value{T}, Nothing}
		}, Nothing}
	backward::Function
	function Value(;
	  data::T,
	  children::Union{
		Tuple{Symbol,
			Value{T},
			Union{Value{T}, Nothing}
		}, Nothing} = nothing,
	  label::Union{String, Nothing}=nothing,
	  backward = nothing,
	) where T<:AbstractFloat
		return new{T}(
			data,
			zero(T),
			label,
			children,
			isnothing(backward) ? () -> zero(T) : backward
		)
	end
end

# ╔═╡ a76c175e-19a9-444f-8590-13e4b57de8c5
function Base.:+(x::Value{T}, y::Value{T}) where T 
    out = Value(
		data=x.data + y.data,
		children=(:+,x,y)
	)
	out.backward = function ()
		x.grad += out.grad
		y.grad += out.grad
	end
    return out
end

# ╔═╡ e6a22cc9-8543-44fe-97de-d76f3354ceb6
function Base.:*(x::Value{T}, y::Value{T}) where T 
    out = Value(
		data=x.data * y.data,
		children=(:*,x,y)
	)
	out.backward = function ()
		x.grad += y.data * out.grad
		y.grad += x.data * out.grad
	end
    return out
end

# ╔═╡ 1f13d05d-f2bc-4134-a344-5352f7e4e27c
function Base.:tanh(x::Value{T}) where T 
    out = Value(
		data=tanh(x.data),
		children=(:tanh,x,nothing)
	)
	out.backward = function ()
		x.grad += (1 - out.data^2) * out.grad
	end
	return out
end

# ╔═╡ 0c957a70-a334-4ded-8131-db1ef41638ca
function toposort(value::V, init_grad=true) where V <: Value{T} where T
	topo = Vector{Value}()
	visited = Set{Value}()
	function build_topo(_::Nothing)
	end
	function build_topo(v)
		if v ∉ visited
			if init_grad
				v.grad = 0
			end
			push!(visited, v)
			if !isnothing(v.children)
			  build_topo(v.children[2])
			  build_topo(v.children[3])
			end
			push!(topo,v)
		end
		topo
	end
	build_topo(value)
	topo
end

# ╔═╡ 6bbaee3d-b446-4647-b057-5c453dd1e5cf
function backward_propagate(value::V) where V <: Value{T} where T
	topo = toposort(value)
	value.grad = 1
	for v in reverse(topo)
		v.backward()
	end
end

# ╔═╡ 9c66b761-fffe-43f4-b8f2-ab0e0b78f291
function build_graph(v::Value, print_string=false)
	graph = """
	strict digraph {
       rankdir=LR
	"""
	graph = build_graph(v, graph)
	graph = graph * "}"
	f = tempname()*".dot"
	write(f, graph)
	print_string && println(graph)
	return GraphViz.load(open(f))
end

# ╔═╡ cb3978fe-7221-4e63-9814-b03150e8c398
function build_graph(v::Value, graph::String)
	if !isnothing(v.children)
		op = v.children[1]
		p = v.children[2]
		q = v.children[3]
		graph = build_graph(p, graph)
		isnothing(q) || (graph = build_graph(q, graph))
		graph = graph * """
		"$(string(hash(p)))" -> "$(string(hash(v)))"
		"""
		if !isnothing(q)
			graph = graph * """
			"$(string(hash(q)))" -> "$(string(hash(v)))"
			"""
		end
		graph = graph * """
		"$(string(hash(v)))" [
	 		label="
		    $(isnothing(v.label) ? "" : string(v.label)*" |")
			$(string(v.children[1])) |
	  		data=$(string(v.data)) |
	  		grad=$(string(v.grad))",
	  		shape="record"]
		"""
	else
		graph = graph * """
		"$(string(hash(v)))" [
		  label="$(string(isnothing(v.label) ? ' ' : v.label)) |
		  data=$(string(v.data)) |
		  grad=$(string(v.grad))",
		  shape="record"]
		"""
	end
	return graph
end

# ╔═╡ 00a4740f-f845-4f1a-995b-3a297cfeb1c9
a = Value(data=-1/4., label="a")

# ╔═╡ 30034c69-2f7a-48c0-92d8-3d161dafa2f8
b = tanh(a);

# ╔═╡ b904f613-6f09-4479-81aa-c912e04a4c13
c = b + a;

# ╔═╡ 2d237e02-5af0-4d08-9e5b-8ae7b2420836
d = tanh(b);

# ╔═╡ e9484c38-1785-4a96-b7fb-1886d2fa737c
e = c * d;

# ╔═╡ 91aa1620-572e-488a-8451-ba3cb4c8b63e
backward_propagate(e)

# ╔═╡ 3c9e1ff6-9ce6-4f05-96fb-70ee40052409
build_graph(e)

# ╔═╡ Cell order:
# ╠═2fc9f05e-b170-4575-8dd4-ac77f64494fc
# ╠═e5e1afe4-43ce-435e-b734-e7321ca8bfd8
# ╠═a76c175e-19a9-444f-8590-13e4b57de8c5
# ╠═e6a22cc9-8543-44fe-97de-d76f3354ceb6
# ╠═1f13d05d-f2bc-4134-a344-5352f7e4e27c
# ╠═0c957a70-a334-4ded-8131-db1ef41638ca
# ╠═6bbaee3d-b446-4647-b057-5c453dd1e5cf
# ╠═9c66b761-fffe-43f4-b8f2-ab0e0b78f291
# ╠═cb3978fe-7221-4e63-9814-b03150e8c398
# ╠═00a4740f-f845-4f1a-995b-3a297cfeb1c9
# ╠═30034c69-2f7a-48c0-92d8-3d161dafa2f8
# ╠═b904f613-6f09-4479-81aa-c912e04a4c13
# ╠═2d237e02-5af0-4d08-9e5b-8ae7b2420836
# ╠═e9484c38-1785-4a96-b7fb-1886d2fa737c
# ╠═91aa1620-572e-488a-8451-ba3cb4c8b63e
# ╠═3c9e1ff6-9ce6-4f05-96fb-70ee40052409
