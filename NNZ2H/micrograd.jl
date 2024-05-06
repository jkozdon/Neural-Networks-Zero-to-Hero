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
	function Value(data::T) where T<:AbstractFloat
		return Value(;data=data)
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

# ╔═╡ 0abb4567-8665-4c4d-b4ea-093d87656e2c
Base.:*(x::Value{T}, y::T) where T = x * Value(y)

# ╔═╡ fb1aa33d-7370-408f-bd85-22cd1da189b7
Base.:*(x::T, y::Value{T}) where T = Value(x) * y

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

# ╔═╡ 4d896a33-68a5-4e9a-bcc1-8a768ec572cf
struct Neuron{T <: AbstractFloat, N}
	w::NTuple{N, Value{T}}
	b::Value{T}
	activation::Function
	function Neuron{T, N}(;activation=tanh) where T <: AbstractFloat where N
		w = ntuple(_->Value(rand(T)), N)
		b = Value(rand(T))
		new{T,N}(w,b,activation)
	end
end

# ╔═╡ 765a1144-5fe5-400b-b02d-080cc7fcce88
function (n::Neuron{T, N})(x) where {T, N}
	@assert(N == length(x))
	n.activation(mapreduce(wx -> wx[1] * wx[2], +, zip(n.w, x)) + n.b)
end

# ╔═╡ 9f0e6e55-285e-4c86-a1ec-e66a37b7331e
struct Layer{T<:AbstractFloat, Nin, Nout}
	data::NTuple{Nout, Neuron{T, Nin}}
	function Layer{T, Nin, Nout}(;activation=tanh) where {T, Nin, Nout}
		new(ntuple(_->Neuron{T, Nin}(;activation=activation), Nout))
	end
end

# ╔═╡ a65f1980-25f1-42e5-ba22-4437604036ed
function (l::Layer{T, Nin, Nout})(x) where {T, Nin, Nout}
	map(n->n(x), l.data)
end

# ╔═╡ b9a6f0d8-d2e8-402c-ba84-bb66f04f3d0c
struct MLP{T<:AbstractFloat, Nin}
	data::Vector{Layer{T}}
	function MLP{T}(d::I...) where {T, I}
		new{T, d[1]}(map(N->Layer{T, N[1], N[2]}(), zip(d, (d[2:end]..., 1))))
	end
end

# ╔═╡ 97379022-055c-4770-afc7-22d56a4d0a25
function (mlp::MLP{T, Nin})(x) where {T, Nin}
	for i in 1:length(mlp.data)
	  x = mlp.data[i](x)
	end
	x[1]
end

# ╔═╡ 18e7b40e-e965-43be-9284-d615cd4d5e57
m = MLP{Float64}(3,4,5)

# ╔═╡ 63fee104-6bb7-4d1a-ad74-9e4b8ce0fade
x = rand(3)

# ╔═╡ e34800e4-7613-44fb-bf95-8d458ce52f49
build_graph(m(x))

# ╔═╡ Cell order:
# ╠═2fc9f05e-b170-4575-8dd4-ac77f64494fc
# ╠═e5e1afe4-43ce-435e-b734-e7321ca8bfd8
# ╠═a76c175e-19a9-444f-8590-13e4b57de8c5
# ╠═e6a22cc9-8543-44fe-97de-d76f3354ceb6
# ╠═0abb4567-8665-4c4d-b4ea-093d87656e2c
# ╠═fb1aa33d-7370-408f-bd85-22cd1da189b7
# ╠═1f13d05d-f2bc-4134-a344-5352f7e4e27c
# ╠═0c957a70-a334-4ded-8131-db1ef41638ca
# ╠═6bbaee3d-b446-4647-b057-5c453dd1e5cf
# ╠═9c66b761-fffe-43f4-b8f2-ab0e0b78f291
# ╠═cb3978fe-7221-4e63-9814-b03150e8c398
# ╠═4d896a33-68a5-4e9a-bcc1-8a768ec572cf
# ╠═765a1144-5fe5-400b-b02d-080cc7fcce88
# ╠═9f0e6e55-285e-4c86-a1ec-e66a37b7331e
# ╠═a65f1980-25f1-42e5-ba22-4437604036ed
# ╠═b9a6f0d8-d2e8-402c-ba84-bb66f04f3d0c
# ╠═97379022-055c-4770-afc7-22d56a4d0a25
# ╠═18e7b40e-e965-43be-9284-d615cd4d5e57
# ╠═63fee104-6bb7-4d1a-ad74-9e4b8ce0fade
# ╠═e34800e4-7613-44fb-bf95-8d458ce52f49
