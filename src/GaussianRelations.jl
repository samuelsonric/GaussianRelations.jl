module GaussianRelations


export CenteredGaussianRelation, CenteredPrecisionForm, CenteredCovarianceForm
export GaussianRelation, PrecisionForm, CovarianceForm
export disintegrate, kleisli, params


using FillArrays
using LinearAlgebra
using StatsAPI
using StatsAPI: params


const DEFAULT_TOLERANCE = 1e-8


abstract type GaussianRelation end


function StatsAPI.params(relation::GaussianRelation, i::Integer)
    params(relation)[i]
end


#=
"""
    params(relation::GaussianRelation)

Get the parameters of a Gaussian relation.
"""
StatsAPI.params(relation::GaussianRelation)


"""
    disintegrate(relation::GaussianRelation, mask::AbstractVector{Bool}; tol::Real=1e-8)

Given a centered Gaussian relation
    f: 0 → m + n,
compute a matrix
     M: m → n
and centered Gaussian relations
     g: 0 → m and h: 0 → n
such that
     f = (g ⊗ h) ; [ I 0 ]
                   [ M I ].
"""
disintegrate(relation::GaussianRelation, mask::AbstractVector{Bool}; tol::Real=1e-8)


"""
    +(relation::GaussianRelation, v::AbstractVector)

Compute the sum
    f + N(v, 0),
where f: 0 → n is a Gaussian relation and v ∈ ℝⁿ is a vector.
"""
Base.:+(relation::GaussianRelation, v::AbstractVector)


"""
    *(M::AbstractMatrix, relation::Union{CovarianceForm, CenteredCovarianceForm})

Compute the composite
    f ; M,
where f: 0 → n is a Gaussian relation in covariance form and M: m → n is a matrix.
"""
Base.:*(M::AbstractMatrix, relation::Union{CovarianceForm, CenteredCovarianceForm})


"""
    *(M::AbstractMatrix, relation::Union{CovarianceForm, CenteredCovarianceForm})

Compute the composite
    f ; M†,
where f: 0 → n is a Gaussian relation in precision form and M: n → m is a matrix.
"""
Base.:*(relation::Union{PrecisionForm, CenteredPrecisionForm}, M::AbstractMatrix)
=#


include("centered.jl")
include("relation.jl")


end
