module GaussianRelations


export CenteredGaussianRelation, CenteredPrecisionForm, CenteredCovarianceForm
export GaussianRelation, PrecisionForm, CovarianceForm
export disintegrate, kleisli, oapply, otimes, params, push, pull


using Catlab.ACSetInterface
using Catlab.FinSets
using Catlab.FinSets: FinDomFunctionVector
using Catlab.Theories
using Catlab.UndirectedWiringDiagrams
using Catlab.WiringDiagramAlgebras
using FillArrays
using LinearAlgebra
using LinearAlgebra: checksquare
using StatsAPI
using StatsAPI: params


const DEFAULT_TOLERANCE = 1e-8


include("centered.jl")
include("relation.jl")


end
