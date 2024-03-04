module GaussianRelations


export CenteredGaussianRelation, CenteredPrecisionForm, CenteredCovarianceForm
export GaussianRelation, PrecisionForm, CovarianceForm
export disintegrate, kleisli, otimes, params


using FillArrays
using LinearAlgebra
using StatsAPI
using StatsAPI: params


const DEFAULT_TOLERANCE = 1e-8


abstract type GaussianRelation end


function StatsAPI.params(relation::GaussianRelation, i::Integer)
    params(relation)[i]
end


include("centered.jl")
include("relation.jl")


end
