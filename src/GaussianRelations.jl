module GaussianRelations


export GaussianRelation, GaussianMap
export disintegrate, marginalize, oapply


using Catlab.ACSetInterface
using Catlab.UndirectedWiringDiagrams
using Catlab.WiringDiagramAlgebras
using CommonSolve
using CommonSolve: solve
using LinearAlgebra


include("saddle_point_problem.jl")
include("gaussian_relation.jl")
include("gaussian_map.jl")


end
