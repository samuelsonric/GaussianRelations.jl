# A factorized matrix of the form
# [ A B ]
# [ B 0 ],
# where A and B are positive semidefinite.
struct SaddlePointProblem{T₁, T₂, T₃, T₄, T₅}
    D₁::T₁
    Q₁::T₂
    Q₂::T₃
    Q₂ᵀ_A_Q₁::T₄
    Q₂ᵀ_A_Q₂::T₅
end


# Factorize a matrix of the form
# [ A B ]
# [ B 0 ],
# where A and B are positive semidefinite.
function SaddlePointProblem(A::AbstractMatrix, B::AbstractMatrix; tol::Real=1e-8)
    D, Q = eigen(B)
    n = count(D .> tol)

    D₁ = Diagonal(D[1:n])
    Q₁ = Q[:, 1:n]
    Q₂ = Q[:, n + 1:end]
    Q₂ᵀ_A_Q₁ = Q₂' * A * Q₁
    Q₂ᵀ_A_Q₂ = Q₂' * A * Q₂

    Q₂ᵀ_A_Q₂ = qr(Q₂ᵀ_A_Q₂, ColumnNorm())
    SaddlePointProblem(D₁, Q₁, Q₂, Q₂ᵀ_A_Q₁, Q₂ᵀ_A_Q₂)
end


# Solve the saddle point problem
# [ A B ] [ l ] = [ a ]
# [ B 0 ] [ λ ] = [ b ],
# where A and B are positive semidefinite.
function CommonSolve.solve(problem::SaddlePointProblem, a, b)
    D₁ = problem.D₁
    Q₁ = problem.Q₁
    Q₂ = problem.Q₂
    Q₂ᵀ_A_Q₁ = problem.Q₂ᵀ_A_Q₁
    Q₂ᵀ_A_Q₂ = problem.Q₂ᵀ_A_Q₂

    u₁ = D₁ \ (Q₁' * b)
    u₂ = Q₂ᵀ_A_Q₂ \ (a - Q₂ᵀ_A_Q₁ * u₁)

    Q₁ * u₁ + Q₂ * u₂
end
