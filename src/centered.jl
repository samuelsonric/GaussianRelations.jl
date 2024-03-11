# A centered Gaussian relation.
struct CenteredGaussianRelation{T, PT <: AbstractMatrix, ST <: AbstractMatrix}
    precision::PT
    support::ST

    function CenteredGaussianRelation{T}(P::PT, S::ST) where {T, PT <: AbstractMatrix, ST <: AbstractMatrix}
        m = checksquare(P)
        n = checksquare(S)
        @assert m == n
        new{T, PT, ST}(P, S)
    end
end


# A centered Gaussian relation in precision form.
const CenteredPrecisionForm = CenteredGaussianRelation{false}


# A centered Gaussian relation in covariance form.
const CenteredCovarianceForm = CenteredGaussianRelation{true}


function CenteredGaussianRelation{T}(relation::CenteredGaussianRelation{T}) where T
    CenteredGaussianRelation{T}(relation.precision, relation.support)
end


function CenteredGaussianRelation{T}(relation::CenteredGaussianRelation) where T
    P, S = params(relation)
    U, V, D = factorizepsd(P + S)
    M = solvespp(D, quad(S, V), I, 0I)
    CenteredGaussianRelation{T}(quad(D, M * V'), quad(I, U'))
end


function CenteredGaussianRelation{T}(P::AbstractMatrix) where T
    n = size(P, 1)
    CenteredGaussianRelation{T}(P, Zeros(n, n))
end


function CenteredGaussianRelation{T, PT, ST}(args...; kwargs...) where {T, PT, ST}
    P, S = params(CenteredGaussianRelation{T}(args...; kwargs...))
    CenteredGaussianRelation{T}(convert(PT, P), convert(ST, S))
end


function Base.length(relation::CenteredGaussianRelation)
    size(relation.precision, 1)
end


function StatsAPI.params(relation::CenteredGaussianRelation, i::Integer)
    @assert 1 <= i <= 2
    params(relation)[i]
end


function StatsAPI.params(relation::CenteredGaussianRelation)
    relation.precision, relation.support
end


function Theories.otimes(left::CenteredGaussianRelation{T}, right::CenteredGaussianRelation{T}) where T
    P₁, S₁ = params(left)
    P₂, S₂ = params(right)
    CenteredGaussianRelation{T}(cat(P₁, P₂, dims=(1, 2)), cat(S₁, S₂, dims=(1, 2)))
end


function pull(M::AbstractMatrix, relation::CenteredGaussianRelation{T}, ::Val{T}) where T
    P, S = params(relation)
    CenteredGaussianRelation{T}(quad(P, M), quad(S, M))
end


function pull(M::AbstractMatrix, relation::CenteredGaussianRelation, ::Val{T}) where T
    CenteredGaussianRelation{!T}(pull(M, CenteredGaussianRelation{T}(relation), Val(T)))
end


function pull(f::FinFunction, relation::CenteredGaussianRelation{T}, ::Val{T}) where T
    epi, mono = epi_mono(f)
    pull_epi(epi, pull_mono(mono, relation, Val(T)), Val(T))
end


function pull(f::FinFunction, relation::CenteredGaussianRelation{T}, ::Val) where T
    f = collect(f)
    P = relation.precision[f, f]
    S = relation.support[f, f]
    CenteredGaussianRelation{T}(P, S) 
end


function pull_epi(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val{T}) where {T, PT, ST}
    m = length(dom(f))
    n = length(codom(f))

    P = zeros(eltype(PT), m, m)
    S = zeros(eltype(ST), m, m)

    section = zeros(Int, n)
    indices = zeros(Int, n)

    for i in 1:m
        v = f(i)

        if iszero(section[v])
            section[v] = i
        else
            j = indices[v]
            S[i, i] += 1
            S[j, j] += 1
            S[i, j] = S[j, i] = -1      
        end

        indices[v] = i
    end

    P[section, section] .+= relation.precision
    S[section, section] .+= relation.support

    CenteredGaussianRelation{T}(P, S)
end


function pull_epi(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val) where {T, PT, ST}
    pull(f, relation, Val(!T))
end


function pull_mono(f::FinFunction, relation::CenteredGaussianRelation{T}, ::Val{T}) where T
    n = length(relation)
    P, S = params(relation)

    i₁ = trues(n)
    i₂ = collect(f)
    i₁[i₂] .= false

    P₁₁, P₁₂, P₂₁, P₂₂ = getblocks(P, i₁, i₂)
    S₁₁, S₁₂, S₂₁, S₂₂ = getblocks(S, i₁, i₂)
    M = solvespp(P₁₁, S₁₁, P₁₂, S₁₂)

    CenteredGaussianRelation{T}(
        P₂₂ + quad(P₁₁, M) - P₂₁ * M - M' * P₁₂, 
        S₂₂ - quad(S₁₁, M))
end


function pull_mono(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val) where {T, PT, ST}
    pull(f, relation, Val(!T))
end


function push(M::AbstractMatrix, relation::CenteredGaussianRelation, ::Val{T}) where T
    pull(M', relation, Val(!T))
end


function push(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val{T}) where {T, PT, ST}
    m = length(dom(f))
    n = length(codom(f))

    P = zeros(eltype(PT), n, n)
    S = zeros(eltype(ST), n, n)

    for i in 1:m, j in 1:m
        P[f(i), f(j)] += relation.precision[i, j]
        S[f(i), f(j)] += relation.support[i, j]
    end

    CenteredGaussianRelation{T}(P, S)
end


function push(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val) where {T, PT, ST}
    epi, mono = epi_mono(f)
    push_mono(mono, push_epi(epi, relation, Val(!T)), Val(!T))
end


function push_epi(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val{T}) where {T, PT, ST}
    push(f, relation, Val(T))
end


function push_epi(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val) where {T, PT, ST}
    m = length(dom(f))
    n = length(codom(f))

    section = zeros(Int, n)
    indices = zeros(Int, n)
    parent  = zeros(Int, m)

    for i in 1:m
        v = f(i)

        if iszero(section[v])
            section[v] = i
        else
            parent[i] = indices[v]
        end

        indices[v] = i
    end

    P = Matrix{eltype(PT)}(undef, m, m)
    S = Matrix{eltype(ST)}(undef, m, m)

    for i₁ in 1:m
        j₁ = parent[i₁]

        for i₂ in i₁:m
            j₂ = parent[i₂]

            p = relation.precision[i₁, i₂]
            s = relation.support[i₁, i₂]

            if !iszero(j₁)
                p -= relation.precision[j₁, i₂]
                s -= relation.support[j₁, i₂]
            end

            if !iszero(j₂)
                p -= relation.precision[i₁, j₂]
                s -= relation.support[i₁, j₂]
            end

            if !iszero(j₁) && !iszero(j₂)
                p += relation.precision[j₁, j₂]
                s += relation.support[j₁, j₂]
            end

            P[i₁, i₂] = p
            S[i₁, i₂] = s
        end
    end

    P = Symmetric(P)
    S = Symmetric(S)
    pull_mono(FinFunction(section, m), CenteredGaussianRelation{T}(P, S), Val(T))
end


function push_mono(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val{T}) where {T, PT, ST}
    push(f, relation, Val(T))
end


function push_mono(f::FinFunction, relation::CenteredGaussianRelation{T, PT, ST}, ::Val) where {T, PT, ST}
    n = length(codom(f))
    indices = collect(f)

    P = zeros(eltype(PT), n, n)
    S = Matrix{eltype(ST)}(I, n, n)

    P[indices, indices] = relation.precision
    S[indices, indices] = relation.support

    CenteredGaussianRelation{T}(P, S)
end


function disintegrate(relation::CenteredGaussianRelation{T}, mask::AbstractVector{Bool}) where T
    i₁ = !T .⊻ mask
    i₂ =  T .⊻ mask
    P, S = params(relation)

    P₁₁, P₁₂, P₂₁, P₂₂ = getblocks(P, i₁, i₂)
    S₁₁, S₁₂, S₂₁, S₂₂ = getblocks(S, i₁, i₂)
    M = solvespp(P₁₁, S₁₁, P₁₂, S₁₂)

    r₁ = CenteredGaussianRelation{T}(P₁₁, S₁₁)

    r₂ = CenteredGaussianRelation{T}(
        P₂₂ + quad(P₁₁, M) - P₂₁ * M - M' * P₁₂, 
        S₂₂ - quad(S₁₁, M))

    T ? (r₁, r₂, -M') : (r₂, r₁, M)
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
    Uᵀ_A_U = qr(quad(A, U), ColumnNorm())

    v = V * (D \ (V' * f))
    u = U * (Uᵀ_A_U \ (U' * (e - A * v)))

    u + v    
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


# Compute
#     Mᵀ A M,
# where A is positive semidefinite.
function quad(A, M::AbstractMatrix)
    Symmetric(M' * A * M)
end


# Factorize a function f: i → k as a composite
#    f = e ; m,
# where e: i → j is a surjection and m: j → k is an injection.
function epi_mono(f::FinFunction)    
    epi = zeros(Int, length(codom(f)))
    mono = zeros(Int, 0)
    
    for i in dom(f)
        j = f(i)
        
        if iszero(epi[j])
            push!(mono, j)
            epi[j] = length(mono)
        end
    end
    
    set = FinSet(length(mono))
    epi = FinDomFunctionVector(epi[collect(f)], dom(f), set)
    mono = FinDomFunctionVector(mono, set, codom(f))
    
    epi, mono
end
