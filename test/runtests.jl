using Catlab, Catlab.Programs
using GaussianRelations
using LinearAlgebra
using Test


function ≅(a, b)
    isapprox(a, b; atol=1e-8)
end


@testset "Positive Definite" begin
    P = [
        5/16 -1/8   0
       -1/8   7/12  1/3
        0     1/3   1/3
    ]

    C = [
        4  2 -2
        2  5 -5
       -2 -5  8
    ]

    M = [
        1  2 -1
        2  1  2
       -1  2  1
    ]

    m = [1, -3, 4]
    v = [3, 9, -1]

    mask = [true, true, false]


    @testset "CenteredPrecisonForm" begin
        relation = CenteredPrecisionForm{Matrix{Float64}, Matrix{Float64}}(P)
        marginal, conditional, matrix = disintegrate(relation, mask)

        @test length(relation) == 3
        @test params(relation, 1) ≅ P
        @test params(kleisli(relation), 1) ≅ P
        @test params(M * relation, 1) ≅ inv(M * C * M')
        @test params(M \ relation, 1) ≅ M' * P * M
        @test params(otimes(relation, relation), 1) ≅ cat(P, P; dims=(1, 2))
        @test params(CenteredCovarianceForm(relation), 1) ≅ C
        @test params(marginal, 1) ≅ inv(C[mask, mask])
        @test params(conditional, 1) ≅ P[.!mask, .!mask]
        @test matrix ≅ -C[.!mask, mask] / C[mask, mask]
    end

    @testset "CenteredCovarianceForm" begin
        relation = CenteredCovarianceForm{Matrix{Float64}, Matrix{Float64}}(C)
        marginal, conditional, matrix = disintegrate(relation, mask)

        @test length(relation) == 3
        @test params(relation, 1) ≅ C
        @test params(kleisli(relation), 1) ≅ C
        @test params(M * relation, 1) ≅ M * C * M'
        @test params(M \ relation, 1) ≅ inv(M' * P * M)
        @test params(otimes(relation, relation), 1) ≅ cat(C, C; dims=(1, 2))
        @test params(CenteredPrecisionForm(relation), 1) ≅ P
        @test params(marginal, 1) ≅ C[mask, mask]
        @test params(conditional, 1) ≅ inv(P[.!mask, .!mask])
        @test matrix ≅ -C[.!mask, mask] / C[mask, mask]
    end


    @testset "PrecisionForm" begin
        relation = PrecisionForm{Matrix{Float64}, Matrix{Float64}}(m, P)
        marginal, conditional, matrix = disintegrate(relation, mask)

        @test length(relation) == 3
        @test cov(relation) ≅ C
        @test mean(relation) ≅ m
        @test params(relation, 1) ≅ P
        @test params(relation, 3) ≅ m
        @test params(v + relation, 1) ≅ P
        @test params(v + relation, 3) ≅ v + m
        @test params(M * relation, 1) ≅ inv(M * C * M')
        @test params(M * relation, 3) ≅ M * m
        @test params(M \ relation, 1) ≅ M' * P * M
        @test params(M \ relation, 3) ≅ inv(M' * P * M) * M' * P * m
        @test params(otimes(relation, relation), 1) ≅ cat(P, P; dims=(1, 2))
        @test params(otimes(relation, relation), 3) ≅ [m; m]
        @test params(CovarianceForm(relation), 1) ≅ C
        @test params(CovarianceForm(relation), 3) ≅ m
        @test params(marginal, 1) ≅ inv(C[mask, mask])
        @test params(marginal, 3) ≅ m[mask]
        @test params(conditional, 1) ≅ P[.!mask, .!mask]
        @test params(conditional, 3) ≅ inv(P[.!mask, .!mask]) * P[.!mask, :] * m
        @test matrix ≅ -C[.!mask, mask] / C[mask, mask]
    end


    @testset "CovarianceForm" begin
        relation = CovarianceForm{Matrix{Float64}, Matrix{Float64}}(m, C)
        marginal, conditional, matrix = disintegrate(relation, mask)

        @test length(relation) == 3
        @test cov(relation) ≅ C
        @test mean(relation) ≅ m
        @test params(relation, 1) ≅ C
        @test params(relation, 3) ≅ m
        @test params(v + relation, 1) ≅ C
        @test params(v + relation, 3) ≅ v + m
        @test params(M * relation, 1) ≅ M * C * M'
        @test params(M * relation, 3) ≅ M * m
        @test params(M \ relation, 1) ≅ inv(M' * P * M)
        @test params(M \ relation, 3) ≅ inv(M' * P * M) * M' * P * m
        @test params(otimes(relation, relation), 1) ≅ cat(C, C; dims=(1, 2))
        @test params(otimes(relation, relation), 3) ≅ [m; m]
        @test params(PrecisionForm(relation), 1) ≅ P
        @test params(PrecisionForm(relation), 3) ≅ m
        @test params(marginal, 1) ≅ C[mask, mask]
        @test params(marginal, 3) ≅ m[mask]
        @test params(conditional, 1) ≅ inv(P[.!mask, .!mask])
        @test params(conditional, 3) ≅ inv(P[.!mask, .!mask]) * P[.!mask, :] * m
        @test matrix ≅ -C[.!mask, mask] / C[mask, mask]
    end
end


@testset "Noisy Resistor" begin
    σ₁ = 1/2
    σ₂ = 2/3
    R₁ = 2
    R₂ = 4
    V₀ = 5

    M = [-1 1; R₂ R₁] / (R₁ + R₂)

    @testset "PrecisionForm" begin
        # ϵ₁ ~ N(0, σ₁²)
        ϵ₁ = PrecisionForm(0, σ₁^-2)

        # ϵ₂ ~ N(0, σ₂²)
        ϵ₂ = PrecisionForm(0, σ₂^-2)

        #         [ I₁ ]
        # [-R₁ 1] [ V₁ ] = ϵ₁
        IV₁ = [-R₁ 1] \ ϵ₁

        @test nullspace(params(IV₁, 1)) ≅ nullspace([-R₁ 1]) || nullspace(params(IV₁, 1)) ≅ -nullspace([-R₁ 1])
        @test all(params([-R₁ 1] * IV₁) .≅ params(ϵ₁))

        #        [ I₁ ]
        # [R₂ 1] [ V₂ ] = ϵ₂ + V₀
        IV₂ = [R₂ 1] \ (ϵ₂ + V₀)


        # [ 1 0 ]         [ I₁ ]
        # [ 0 1 ]         [ V₁ ]
        # [ 1 0 ] [ I ]   [ I₂ ]
        # [ 0 1 ] [ V ] = [ V₂ ]
        IV = [1 0; 0 1; 1 0; 0 1] \ otimes(IV₁, IV₂)

        @test params(IV, 1) ≅ inv(M * [σ₁^2 0; 0 σ₂^2] * M')
        @test params(IV, 2) ≅ [0 0; 0 0]
        @test params(IV, 3) ≅  M * [0; V₀]

        #       IV₁
        #      /   \
        # --- I     V ---
        #      \   /
        #       IV₂
        diagram = @relation (I, V) begin
            IV₁(I, V)
            IV₂(I, V)
        end

        IV = oapply(diagram, Dict(:IV₁ => IV₁, :IV₂ => IV₂), Dict(:I => 1, :V => 1))

        @test params(IV, 1) ≅ inv(M * [σ₁^2 0; 0 σ₂^2] * M')
        @test params(IV, 2) ≅ [0 0; 0 0]
        @test params(IV, 3) ≅  M * [0; V₀]
    end

    @testset "CovarianceForm" begin
        # ϵ₁ ~ N(0, σ₁²)
        ϵ₁ = CovarianceForm(0, σ₁^2)

        # ϵ₂ ~ N(0, σ₂²)
        ϵ₂ = CovarianceForm(0, σ₂^2)

        #         [ I₁ ]
        # [-R₁ 1] [ V₁ ] = ϵ₁
        IV₁ = [-R₁ 1] \ ϵ₁

        @test nullspace(params(IV₁, 2)) ≅ nullspace([1 R₁]) || nullspace(params(IV₁, 2)) ≅ -nullspace([1 R₁])
        @test all(params([-R₁ 1] * IV₁) .≅ params(ϵ₁))

        #        [ I₁ ]
        # [R₂ 1] [ V₂ ] = ϵ₂ + V₀
        IV₂ = [R₂ 1] \ (ϵ₂ + V₀)

        # [ 1 0 ]         [ I₁ ]
        # [ 0 1 ]         [ V₁ ]
        # [ 1 0 ] [ I ]   [ I₂ ]
        # [ 0 1 ] [ V ] = [ V₂ ]
        IV = [1 0; 0 1; 1 0; 0 1] \ otimes(IV₁, IV₂)

        @test params(IV, 1) ≅ M * [σ₁^2 0; 0 σ₂^2] * M'
        @test params(IV, 2) ≅ [0 0; 0 0]
        @test params(IV, 3) ≅  M * [0; V₀]

        #       IV₁
        #      /   \
        # --- I     V ---
        #      \   /
        #       IV₂
        diagram = @relation (I, V) begin
            IV₁(I, V)
            IV₂(I, V)
        end

        IV = oapply(diagram, Dict(:IV₁ => IV₁, :IV₂ => IV₂), Dict(:I => 1, :V => 1))

        @test params(IV, 1) ≅ M * [σ₁^2 0; 0 σ₂^2] * M'
        @test params(IV, 2) ≅ [0 0; 0 0]
        @test params(IV, 3) ≅  M * [0; V₀]
    end
end
