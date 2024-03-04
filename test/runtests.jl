using GaussianRelations
using Test


@testset "CenteredGaussianRelation" begin
    Ω = [
        5/16 -1/8   0
       -1/8   7/12  1/3
        0     1/3   1/3
    ]

    Σ = [
        4  2 -2
        2  5 -5
       -2 -5  8
    ]

    relation = CenteredPrecisionForm(Ω)

    @test params(relation, 1) ≈ Ω
    @test params(kleisli(relation), 1) ≈ Ω
    @test params(relation + [1, 2, 3], 1) ≈ Ω
    @test params(relation + [1, 2, 3], 3) ≈ Ω * [1, 2, 3]
    @test params([1 2 3; 4 5 6]' \ relation, 1) ≈ [1 2 3; 4 5 6] * Ω * [1 2 3; 4 5 6]'
    @test params(otimes(relation, relation), 1) ≈ cat(Ω, Ω; dims=(1, 2))
    @test params(CenteredCovarianceForm(relation), 1) ≈ Σ

    marginal, conditional, M = disintegrate(relation, [true, true, false])

    @test params(marginal, 1) ≈ [5/16 -1/8; -1/8 1/4]
    @test params(conditional, 1) ≈ [1/3;;]
    @test M ≈ [0 1]

    relation = CenteredCovarianceForm(Σ)

    @test params(relation, 1) ≈ Σ
    @test params(kleisli(relation), 1) ≈ Σ
    @test params(relation + [1, 2, 3], 1) ≈ Σ
    @test params(relation + [1, 2, 3], 3) ≈ [1, 2, 3]
    @test params([1 2 3; 4 5 6] * relation, 1) ≈ [1 2 3; 4 5 6] * Σ * [1 2 3; 4 5 6]'
    @test params(otimes(relation, relation), 1) ≈ cat(Σ, Σ; dims=(1, 2))
    @test params(CenteredPrecisionForm(relation), 1) ≈ Ω

    marginal, conditional, M = disintegrate(relation, [true, true, false])

    @test params(marginal, 1) ≈ [4 2; 2 5]
    @test params(conditional, 1) ≈ [3;;]
    @test M ≈ [0 1]
end


@testset "GaussianRelation" begin
    Ω = [
        5/16 -1/8   0
       -1/8   7/12  1/3
        0     1/3   1/3
    ]

    Σ = [
        4  2 -2
        2  5 -5
       -2 -5  8
    ]

    μ = [1, -3, 4]

    relation = PrecisionForm(μ, Ω)

    @test params(relation, 1) ≈ Ω
    @test params(relation, 3) ≈ Ω * μ
    @test params(relation + [1, 2, 3], 1) ≈ Ω
    @test params(relation + [1, 2, 3], 3) ≈ Ω * (μ + [1, 2, 3])
    @test params([1 2 3; 4 5 6]' \ relation, 1) ≈ [1 2 3; 4 5 6] * Ω * [1 2 3; 4 5 6]'
    @test params([1 2 3; 4 5 6]' \ relation, 3) ≈ [1 2 3; 4 5 6] * Ω * μ
    @test params(otimes(relation, relation), 1) ≈ cat(Ω, Ω; dims=(1, 2))
    @test params(otimes(relation, relation), 3) ≈ [Ω * μ; Ω * μ]
    @test params(CovarianceForm(relation), 1) ≈ Σ
    @test params(CovarianceForm(relation), 3) ≈ μ

    marginal, conditional, M = disintegrate(relation, [true, true, false])

    @test params(marginal, 1) ≈ [5/16 -1/8; -1/8 1/4]
    @test params(marginal, 3) ≈ [5/16 -1/8; -1/8 1/4] * [1, -3]
    @test params(conditional, 1) ≈ [1/3;;]
    @test params(conditional, 3) ≈ [1/3;;] * [1]
    @test M ≈ [0 1]

    relation = CovarianceForm(μ, Σ)

    @test params(relation, 1) ≈ Σ
    @test params(relation, 3) ≈ μ
    @test params(relation + [1, 2, 3], 1) ≈ Σ
    @test params(relation + [1, 2, 3], 3) ≈ μ + [1, 2, 3]
    @test params([1 2 3; 4 5 6] * relation, 1) ≈ [1 2 3; 4 5 6] * Σ * [1 2 3; 4 5 6]'
    @test params([1 2 3; 4 5 6] * relation, 3) ≈ [1 2 3; 4 5 6] * μ
    @test params(otimes(relation, relation), 1) ≈ cat(Σ, Σ; dims=(1, 2))
    @test params(otimes(relation, relation), 3) ≈ [μ; μ]
    @test params(PrecisionForm(relation), 1) ≈ Ω
    @test params(PrecisionForm(relation), 3) ≈ Ω * μ

    marginal, conditional, M = disintegrate(relation, [true, true, false])

    @test params(marginal, 1) ≈ [4 2; 2 5]
    @test params(marginal, 3) ≈ [1, -3]
    @test params(conditional, 1) ≈ [3;;]
    @test params(conditional, 3) ≈ [1]
    @test M ≈ [0 1]
end
