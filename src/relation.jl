# A Gaussian relaton in precision form.
struct PrecisionForm{T₁, T₂} <: GaussianRelation
    inner::CenteredPrecisionForm{T₁, T₂}
end


# A Gaussian relation in covariance form.
struct CovarianceForm{T₁, T₂} <: GaussianRelation
    inner::CenteredCovarianceForm{T₁, T₂}
end


function PrecisionForm(μ::Real, ω²::Real)
    PrecisionForm([μ], [ω²;;])
end


# Construct the Gaussian relation
#     N(μ, Ω⁻¹)
# in precision form.
function PrecisionForm(μ::AbstractVector, Ω::AbstractMatrix)
    kleisli(CenteredPrecisionForm(Ω)) + μ
end


# Construct the precision form of a Gaussian relation.
function PrecisionForm(relation::PrecisionForm)
    PrecisionForm(relation.inner)
end


# Construct the precision form of a Gaussian relation.
function PrecisionForm(relation::CovarianceForm; tol::Real=DEFAULT_TOLERANCE)
    inner = CenteredPrecisionForm(relation.inner; tol)
    PrecisionForm(inner)
end


function CovarianceForm(μ::Real, σ²::Real)
    CovarianceForm([μ], [σ²;;])
end


# Construct the Gaussian relation
#     N(μ, Σ)
# in covariance form.
function CovarianceForm(μ::AbstractVector, Σ::AbstractMatrix)
    kleisli(CenteredCovarianceForm(Σ)) + μ
end


# Construct the covariance form of a Gaussian relation.
function CovarianceForm(relation::PrecisionForm; tol::Real=DEFAULT_TOLERANCE)
    inner = CenteredCovarianceForm(relation.inner; tol)
    CovarianceForm(inner)
end


# Construct the covariance form of a Gaussian relation.
function CovarianceForm(relation::CovarianceForm)
    CovarianceForm(relation.inner)
end


# Get the arity of a Gaussian relation.
function Base.length(relation::PrecisionForm)
    length(relation.inner) - 1
end


# Get the arity of a Gaussian relation.
function Base.length(relation::CovarianceForm)
    length(relation.inner) - 1
end


# Get the parameters of a Gaussian relation.
function StatsAPI.params(relation::PrecisionForm)
    Ω, S = params(relation.inner)
    Ω[2:end, 2:end], S[2:end, 2:end], Ω[2:end, 1], S[2:end, 1] 
end


# Get the parameters of a Gaussian relation.
function StatsAPI.params(relation::CovarianceForm; tol::Real=DEFAULT_TOLERANCE)
    Σ, F = params(relation.inner)
    @assert Σ[1, 1] > tol || F[1, 1] > tol

    if F[1, 1] > tol
        μ = F[2:end, 1] / -F[1, 1]
        Σ = Σ[2:end, 2:end] + Σ[1, 1] / F[1, 1] * F[2:end, 2:end] + Σ[2:end, 1] * μ' + μ * Σ[1, 2:end]'
        F = F[2:end, 2:end] - F[1, 1] * μ * μ'
    else
        μ = Σ[2:end, 1] / -Σ[1, 1]
        Σ = Σ[2:end, 2:end] - Σ[1, 1] * μ * μ'
        F = F[2:end, 2:end]
    end
    
    Σ, F, μ
end


function kleisli(relation::CenteredPrecisionForm)
    n = length(relation)
    Ω = [0 Zeros(n)'; Zeros(n) relation.Ω]
    S = [0 Zeros(n)'; Zeros(n) relation.S]
    PrecisionForm(CenteredPrecisionForm(Ω, S))
end


function kleisli(relation::CenteredCovarianceForm)
    n = length(relation)
    Σ = [1 Zeros(n)'; Zeros(n) relation.Σ]
    F = [1 Zeros(n)'; Zeros(n) relation.F]
    CovarianceForm(CenteredCovarianceForm(Σ, F))
end


function otimes(left::PrecisionForm, right::PrecisionForm)
    m = length(left)
    n = length(right)

    M = [
        1               Zeros(1, m)     Zeros(1, n)
        Zeros(m, 1)     Eye(m)          Zeros(m, n)
        1               Zeros(1, m)     Zeros(1, n)
        Zeros(n, 1)     Zeros(n, m)     Eye(n)
    ]

    PrecisionForm(M \ otimes(left.inner, right.inner))
end


function otimes(left::CovarianceForm, right::CovarianceForm)
    CovarianceForm(otimes(PrecisionForm(left), PrecisionForm(right)))
end


function Base.:\(M::AbstractMatrix, relation::PrecisionForm)
    m, n = size(M)
    PrecisionForm([1 Zeros(n)'; Zeros(m) M] \ relation.inner)
end


function Base.:\(M::AbstractMatrix, relation::CovarianceForm)
    CovarianceForm(M \ PrecisionForm(relation))
end


function Base.:*(M::AbstractMatrix, relation::PrecisionForm)
    PrecisionForm(M * CovarianceForm(relation))
end


function Base.:*(M::AbstractMatrix, relation::CovarianceForm)
    m, n = size(M)
    CovarianceForm([1 Zeros(n)'; Zeros(m) M] * relation.inner)
end


# Compute the sum
#     ψ + N(v, 0),
# where ψ is a Gaussian relation in precision form.
function Base.:+(relation::PrecisionForm, v::AbstractVector)
    n = length(v)
    M = [1 Zeros(n)'; v Eye(n)]
    PrecisionForm(M \ relation.inner)
end


# Compute the sum
#     ψ + N(v, 0),
# where ψ is a Gaussian relation in covariance form.
function Base.:+(relation::CovarianceForm, v::AbstractVector)
    n = length(v)
    M = [1 Zeros(n)'; -v Eye(n)]
    CovarianceForm(M * relation.inner)
end


# Compute the sum
#     ψ + N(v, 0),
# where ψ is a centered Gaussian relation.
function Base.:+(relation::CenteredGaussianRelation, v::AbstractVector)
    kleisli(relation) + v
end


function Base.:+(relation::GaussianRelation, v::Real)
    n = length(relation)
    relation + fill(v, n)
end


# Given a Gaussian relation
#     f: 0 → m + n,
# compute a matrix
#     M: m → n
# and Gaussian relations
#     g: 0 → m and h: 0 → n
# such that
#     f = (g ⊗ h) ; [ I 0 ]
#                   [ M I ].
function disintegrate(relation::PrecisionForm, mask::AbstractVector{Bool}; tol::Real=DEFAULT_TOLERANCE)
    marginal, conditional, M = disintegrate(relation.inner, [true; mask]; tol)

    m = M[:, 1]
    M = M[:, 2:end]

    marginal = PrecisionForm(marginal)
    conditional = conditional + m
    marginal, conditional, M
end


# Given a Gaussian relation
#     f: 0 → m + n,
# compute a matrix
#     M: m → n
# and Gaussian relations
#     g: 0 → m and h: 0 → n
# such that
#     f = (g ⊗ h) ; [ I 0 ]
#                   [ M I ].
function disintegrate(relation::CovarianceForm, mask::AbstractVector{Bool}; tol::Real=DEFAULT_TOLERANCE)
    marginal, conditional, M = disintegrate(relation.inner, [true; mask]; tol)

    m = M[:, 1]
    M = M[:, 2:end]

    marginal = CovarianceForm(marginal)
    conditional = conditional + m
    marginal, conditional, M
end
