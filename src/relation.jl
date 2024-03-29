# A Gaussian relation.
struct GaussianRelation{T, PT, ST}
    inner::CenteredGaussianRelation{T, PT, ST}

    function GaussianRelation(inner::CenteredGaussianRelation{T, PT, ST}) where {T, PT, ST}
        @assert length(inner) >= 1
        new{T, PT, ST}(inner)
    end
end


# A Gaussian relation in precision form.
const PrecisionForm = GaussianRelation{false}


# A Gaussian relation in covariance form.
const CovarianceForm = GaussianRelation{true}


function GaussianRelation{T}(μ::Real, p²::Real) where T
    GaussianRelation{T}(μ, p², 0)
end


function GaussianRelation{T}(μ::Real, p²::Real, s²::Real) where T
    GaussianRelation{T}([μ], [p²;;], [s²;;])
end


function GaussianRelation{T}(μ::AbstractVector, P::AbstractMatrix) where T
    n = length(μ)
    GaussianRelation{T}(μ, P, Zeros(n, n))
end


function GaussianRelation{T}(μ::AbstractVector, P::AbstractMatrix, S::AbstractMatrix) where T
    kleisli(CenteredGaussianRelation{T}(P, S)) + μ
end


function GaussianRelation{T}(inner::CenteredGaussianRelation) where T
    GaussianRelation(CenteredGaussianRelation{T}(inner))
end


function GaussianRelation{T}(relation::GaussianRelation) where T
    GaussianRelation(CenteredGaussianRelation{T}(relation.inner))
end


function GaussianRelation{T, PT, ST}(args...; kwargs...) where {T, PT, ST}
    inner = GaussianRelation{T}(args...; kwargs...).inner
    GaussianRelation(CenteredGaussianRelation{T, PT, ST}(inner))
end 


function Base.convert(::Type{GaussianRelation{T, PT, ST}}, relation::GaussianRelation{T}) where {T, PT, ST}
    GaussianRelation{T, PT, ST}(relation)
end


function Base.length(relation::GaussianRelation)
    length(relation.inner) - 1
end


function StatsAPI.params(relation::GaussianRelation, i::Integer)
    @assert 1 <= i <= 3
    params(relation)[i]
end


function StatsAPI.params(relation::GaussianRelation)
    n = length(relation)
    marginal, conditional, M = disintegrate(relation.inner, [true; Falses(n)])
    params(conditional)..., reshape(M, n)
end


function Statistics.cov(relation::CovarianceForm)
    params(relation, 1)
end


function Statistics.cov(relation::PrecisionForm)
    cov(CovarianceForm(relation))
end


function Statistics.mean(relation::GaussianRelation)
    params(relation, 3)
end


function kleisli(relation::CenteredGaussianRelation{T}) where T
    P, S = params(relation)
    P = cat(T, P; dims=(1, 2))
    S = cat(T, S; dims=(1, 2))
    GaussianRelation(CenteredGaussianRelation{T}(P, S))
end


function Theories.otimes(left::GaussianRelation, right::GaussianRelation)
    m = length(left.inner)
    n = length(right.inner)
    f = FinDomFunctionVector([1:m; 1; m + 1:m + n - 1], FinSet(m + n), FinSet(m + n - 1))
    GaussianRelation(push_epi(f, otimes(left.inner, right.inner), Val(false)))
end


function Theories.otimes(relations::GaussianRelation{T, PT, ST}...) where {T, PT, ST}
    init = GaussianRelation(CenteredGaussianRelation{T, PT, ST}([T;;], [T;;]))
    reduce(otimes, relations; init)
end


function Base.:+(relation::PrecisionForm, v::AbstractVector)
    n = length(v)
    GaussianRelation(pull([1 Zeros(n)'; v Eye(n)], relation.inner, Val(false)))
end


function Base.:+(relation::CovarianceForm, v::AbstractVector)
    n = length(v)
    GaussianRelation(push([1 Zeros(n)'; -v Eye(n)], relation.inner, Val(false)))
end


function Base.:+(relation::GaussianRelation, v::Real)
    relation + [v]
end


function Base.:+(v, relation::GaussianRelation)
    relation + v
end


function Base.:\(M, relation::Union{GaussianRelation, CenteredGaussianRelation})
    pull(M, relation, Val(false))
end


function Base.:*(M, relation::Union{GaussianRelation, CenteredGaussianRelation})
    push(M, relation, Val(false))
end


function disintegrate(relation::GaussianRelation, mask::AbstractVector{Bool})
    marginal, conditional, M = disintegrate(relation.inner, [true; mask])
    GaussianRelation(marginal), kleisli(conditional) + M[:, 1], M[:, 2:end]
end

function pull(M::AbstractMatrix, relation::GaussianRelation, ::Val{T}) where T
    GaussianRelation(pull(cat(1, M; dims=(1, 2)), relation.inner, Val(T)))
end


function pull(f::FinFunction, relation::GaussianRelation, ::Val{T}) where T
    GaussianRelation(pull(oplus(FinFunction([1], 1), f), relation.inner, Val(T)))
end


function push(M::AbstractMatrix, relation::GaussianRelation, ::Val{T}) where T
    GaussianRelation(push(cat(1, M; dims=(1, 2)), relation.inner, Val(T)))
end


function push(f::FinFunction, relation::GaussianRelation, ::Val{T}) where T
    GaussianRelation(push(oplus(FinFunction([1], 1), f), relation.inner, Val(T)))
end


function WiringDiagramAlgebras.oapply(
    diagram::UndirectedWiringDiagram,
    hom_map::AbstractDict{<:Any, <:GaussianRelation},
    ob_map::AbstractDict{<:Any, <:Integer},
    ::Val{T}=Val(false);
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:variable) where T

    homs = map(diagram[hom_attr]) do attr
        hom_map[attr]
    end

    obs = map(diagram[ob_attr]) do attr
        ob_map[attr]
    end

    oapply(diagram, homs, obs, Val(T))
end


function WiringDiagramAlgebras.oapply(
    diagram::UndirectedWiringDiagram,
    homs::AbstractVector{<:GaussianRelation},
    obs::AbstractVector{<:Integer},
    ::Val{T}=Val(false)) where T

    q_stop = cumsum(obs[diagram[:outer_junction]])
    p_stop = cumsum(obs[diagram[:junction]])
    j_stop = cumsum(obs)
    
    q_start = q_stop .- obs[diagram[:outer_junction]] .+ 1
    p_start = p_stop .- obs[diagram[:junction]] .+ 1
    j_start = j_stop .- obs .+ 1

    q_map = Vector{Int}(undef, q_stop[end])
    p_map = Vector{Int}(undef, p_stop[end])

    for q in parts(diagram, :OuterPort)
        j = diagram[q, :outer_junction]
        q_map[q_start[q]:q_stop[q]] .= j_start[j]:j_stop[j]
    end

    for p in parts(diagram, :Port)
        j = diagram[p, :junction]
        p_map[p_start[p]:p_stop[p]] .= j_start[j]:j_stop[j]
    end

    q_map = FinFunction(q_map, j_stop[end])
    p_map = FinFunction(p_map, j_stop[end])
    relation = otimes(homs...)

    pull(q_map, push(p_map, relation, Val(T)), Val(T))
end
