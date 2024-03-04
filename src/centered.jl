# A centered Gaussian relation.
abstract type CenteredGaussianRelation <: GaussianRelation end


# A centered Gaussian relation in precision form.
struct CenteredPrecisionForm{T₁ <: AbstractMatrix, T₂ <: AbstractMatrix} <: CenteredGaussianRelation
    Ω::T₁
    S::T₂
end


# A centered Gaussian relation in covariance form.
struct CenteredCovarianceForm{T₁ <: AbstractMatrix, T₂ <: AbstractMatrix} <: CenteredGaussianRelation
    Σ::T₁
    F::T₂
end


# Construct the precision form of a centered Gaussian relation.
function CenteredPrecisionForm(relation::CenteredPrecisionForm)
    CenteredPrecisionForm(relation.Ω, relation.S)
end


# Construct the precision form of a centered Gaussian relation.
function CenteredPrecisionForm(relation::CenteredCovarianceForm; tol::Real=DEFAULT_TOLERANCE)
    dual(CenteredCovarianceForm(dual(relation); tol))
end


# Construct the centered Gaussian relation
#     N(0, Ω⁻¹)
# in precision form.
function CenteredPrecisionForm(Ω::AbstractMatrix)
    n = size(Ω, 1)
    CenteredPrecisionForm(Ω, Zeros(n, n))
end


# Construct the covariance form of a centered Gaussian relation.
function CenteredCovarianceForm(relation::CenteredPrecisionForm; tol::Real=DEFAULT_TOLERANCE)
    Ω, S = params(relation)
    U, V, D = factorizepsd(Ω + S; tol)
    M = solvespp(D, V' * S * V, I, 0I; tol)
    
    Σ = V * M' * D * M * V'
    F = U * U'
    CenteredCovarianceForm(Σ, F)
end


# Construct the covariance form of centered Gaussian relation.
function CenteredCovarianceForm(relation::CenteredCovarianceForm)
    CenteredCovarianceForm(relation.Σ, relation.F)
end


# Construct the centered Gaussian relation
#     N(0, Σ)
# in covariance form.
function CenteredCovarianceForm(Σ::AbstractMatrix)
    n = size(Σ, 1)
    CenteredCovarianceForm(Σ, Zeros(n, n))
end


# Get the arity of a centered Gaussian relation.
function Base.length(relation::CenteredPrecisionForm)
    size(relation.Ω, 1)
end


# Get the arity of a centered Gaussian relation.
function Base.length(relation::CenteredCovarianceForm)
    length(dual(relation))
end


# Get the parameters of a centered Gaussian relation.
function StatsAPI.params(relation::CenteredPrecisionForm)
    relation.Ω, relation.S
end


# Get the parameters of a centered Gaussian relation.
function StatsAPI.params(relation::CenteredCovarianceForm)
    relation.Σ, relation.F
end


# Compute the dual of a centered Gaussian relation.
function dual(relation::CenteredPrecisionForm)
    Ω, S = params(relation)
    CenteredCovarianceForm(Ω, S)
end


# Compute the dual of a centered Gaussian relation.
function dual(relation::CenteredCovarianceForm)
    Σ, F = params(relation)
    CenteredPrecisionForm(Σ, F)
end


function otimes(left::CenteredPrecisionForm, right::CenteredPrecisionForm)
    Ω₁, S₁ = params(left)
    Ω₂, S₂ = params(right)
    Ω = cat(Ω₁, Ω₂, dims=(1, 2))
    S = cat(S₁, S₂, dims=(1, 2))
    CenteredPrecisionForm(Ω, S)
end


function otimes(left::CenteredCovarianceForm, right::CenteredCovarianceForm)
    dual(otimes(dual(left), dual(right)))
end


# Compute the composite
#     (f ; M†): 0 → m
# where f: 0 → n is a centered Gaussian relation and M: m → n. is a matrix.
function Base.:\(M::AbstractMatrix, relation::CenteredPrecisionForm)
    Ω, S = params(relation)
    CenteredPrecisionForm(M' * Ω * M, M' * S * M)
end


# Compute the composite
#     (f ; M†): 0 → m
# where f: 0 → n is a centered Gaussian relation and M: m → n. is a matrix.
function Base.:\(M::AbstractMatrix, relation::CenteredCovarianceForm)
    CenteredCovarianceForm(M \ CenteredPrecisionForm(relation))
end


# Compute the composite
#     (f ; M): 0 → n
# where f: 0 → m is a centered Gaussian relation and M: m → n is a matrix.
function Base.:*(M::AbstractMatrix, relation::CenteredPrecisionForm)
    CenteredPrecisionForm(M * CenteredCovarianceForm(relation))
end


# Compute the composite
#     (f ; M): 0 → n
# where f: 0 → m is a centered Gaussian relation and M: m → n is a matrix.
function Base.:*(M::AbstractMatrix, relation::CenteredCovarianceForm)
    dual(M' \ dual(relation))
end


# Given a centered Gaussian relation
#     f: 0 → m + n,
# compute a matrix
#     M: m → n
# and centered Gaussian relations
#     g: 0 → m and h: 0 → n
# such that
#     f = (g ⊗ h) ; [ I 0 ]
#                   [ M I ].
function disintegrate(relation::CenteredPrecisionForm, mask::AbstractVector{Bool}; tol::Real=DEFAULT_TOLERANCE)
    i₁ = .!mask
    i₂ =   mask
    Ω, S = params(relation)

    Ω₁₁, Ω₁₂, Ω₂₁, Ω₂₂ = getblocks(Ω, i₁, i₂)
    S₁₁, S₁₂, S₂₁, S₂₂ = getblocks(S, i₁, i₂)
    M = solvespp(Ω₁₁, S₁₁, Ω₁₂, S₁₂; tol)

    marginal = CenteredPrecisionForm(
        Ω₂₂ + M' * Ω₁₁ * M - Ω₂₁ * M - M' * Ω₁₂, 
        S₂₂ - M' * S₁₁ * M)

    conditional = CenteredPrecisionForm(Ω₁₁, S₁₁)

    marginal, conditional, M
end


# Given a centered Gaussian relation
#     f: 0 → m + n,
# compute a matrix
#     M: m → n
# and centered Gaussian relations
#     g: 0 → m and h: 0 → n
# such that
#     f = (g ⊗ h) ; [ I 0 ]
#                   [ M I ].
function disintegrate(relation::CenteredCovarianceForm, mask::AbstractVector{Bool}; tol::Real=DEFAULT_TOLERANCE)
    marginal, conditional, M = disintegrate(dual(relation), .!mask)
    dual(conditional), dual(marginal), -M'
end


# Compute matrices M₁₁, M₁₂, M₂₁, and M₂₂ such that
#     M = [ M₁₁ M₁₂ ]
#         [ M₂₁ M₂₂ ],
# where M is a square matrix.
function getblocks(M::AbstractMatrix, i₁::AbstractVector, i₂::AbstractVector)
    M₁₁ = M[i₁, i₁]
    M₁₂ = M[i₁, i₂]
    M₂₁ = M[i₂, i₁]
    M₂₂ = M[i₂, i₂]

    M₁₁, M₁₂, M₂₁, M₂₂
end


# Solve the saddle point problem
#     [ A B ] [ x ] = [ e ]
#     [ B 0 ] [ y ] = [ f ],
# where A and B are positive semidefinite, e ∈ col A, and f ∈ col B.
function solvespp(A::AbstractMatrix, B::AbstractMatrix, e, f; tol::Real=DEFAULT_TOLERANCE)
    U, V, D = factorizepsd(B; tol)
    Uᵀ_A_U = qr(U' * A * U, ColumnNorm())

    f = V' * f
    v = D \ f

    e = U' * (e - A * V * v)
    u = Uᵀ_A_U \ e
    
    U * u + V * v
end


# Compute a diagonal matrix
#     D ≻ 0
# and unitary matrix
#     Q = [ U V ]
# such that
#     A =         [ 0 0 ] [ Uᵀ ]
#         [ U V ] [ 0 D ] [ Vᵀ ],
# where A is positive semidefinite.
function factorizepsd(A::AbstractMatrix; tol::Real=DEFAULT_TOLERANCE)
    D, Q = eigen(A)
    n = count(D .< tol)

    U = Q[:, 1:n]
    V = Q[:, n + 1:end]
    D  = Diagonal(D[n + 1:end])

    U, V, D
end
