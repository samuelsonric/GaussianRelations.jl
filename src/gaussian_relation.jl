# A Gaussian relation.
struct GaussianRelation{
    M₁ <: AbstractMatrix,
    M₂ <: AbstractMatrix,
    V₁ <: AbstractVector,
    V₂ <: AbstractVector,
    S <: Real}

    A::M₁
    B::M₂
    a::V₁
    b::V₂
    β::S
end


function GaussianRelation{M₁, M₂, V₁, V₂, S}(relation::GaussianRelation) where {M₁, M₂, V₁, V₂, S}
    A = relation.A
    B = relation.B
    a = relation.a
    b = relation.b
    β = relation.β
    GaussianRelation{M₁, M₂, V₁, V₂, S}(A, B, a, b, β)
end


function Base.length(relation::GaussianRelation)
    size(relation.A, 1)
end
