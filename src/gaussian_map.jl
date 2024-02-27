struct GaussianMap{G <: GaussianRelation, M <: AbstractMatrix}
    relation::G
    matrix::M
end


function disintegrate(relation::GaussianRelation, mask::AbstractVector{Bool})
    i₁ = .!mask
    i₂ =   mask

    A₁₁, A₁₂, A₂₁, A₂₂ = getblocks(relation.A, i₁, i₂)
    B₁₁, B₁₂, B₂₁, B₂₂ = getblocks(relation.B, i₁, i₂)
    a₁, a₂ = getblocks(relation.a, i₁, i₂)
    b₁, b₂ = getblocks(relation.b, i₁, i₂)
    β = relation.β

    problem = SaddlePointProblem(A₁₁, B₁₁)
    L = solve(problem, A₁₂, B₁₂)    
    l = solve(problem, a₁,  b₁)

    marginal = GaussianRelation(
        A₂₂ + L' * A₁₁ * L - A₂₁ * L - L' * A₁₂, 
        B₂₂ - L' * B₁₁ * L,
        a₂  + L' * A₁₁ * l - A₂₁ * l - L' * a₁,
        b₂  - L' * B₁₁ * l,
        β   - l' * B₁₁ * l)

    conditional = GaussianMap(
        GaussianRelation(A₁₁, B₁₁, a₁, b₁, l' * b₁),
        -L)

    marginal, conditional
end


function marginalize(relation::GaussianRelation, mask::AbstractVector)
    marginal, _ = disintegrate(relation, mask)
    marginal
end


function getblocks(M::AbstractMatrix, i₁::AbstractVector, i₂::AbstractVector)
    M₁₁ = M[i₁, i₁]
    M₁₂ = M[i₁, i₂]
    M₂₁ = M[i₂, i₁]
    M₂₂ = M[i₂, i₂]

    M₁₁, M₁₂, M₂₁, M₂₂
end


function getblocks(v::AbstractVector, i₁::AbstractVector, i₂::AbstractVector)
    v₁ = v[i₁]
    v₂ = v[i₂]

    v₁, v₂
end


function WiringDiagramAlgebras.oapply(
    diagram::UndirectedWiringDiagram,   
    generators::AbstractDict{<:Any, <:GaussianRelation})

    generators = map(diagram[:, :name]) do name
        generators[name]
    end

    oapply(diagram, generators)
end


function WiringDiagramAlgebras.oapply(
    diagram::UndirectedWiringDiagram,   
    generators::AbstractVector{<:GaussianRelation})

    n = nparts(diagram, :Junction)
    m = nparts(diagram, :OuterPort)

    A′ = zeros(n, n)
    B′ = zeros(n, n)
    a′ = zeros(n)
    b′ = zeros(n)
    β′ = 0.

    A = zeros(m, m)
    B = zeros(m, m)
    a = zeros(m)
    b = zeros(m)
    β = 0.

    # merge and create
    for b in parts(diagram, :Box)
        ports = incident(diagram, b, :box)
        junctions = diagram[ports, :junction]
        generator = generators[b]

        A′[junctions, junctions] .+= generator.A
        B′[junctions, junctions] .+= generator.B
        a′[junctions] .+= generator.a
        b′[junctions] .+= generator.b
        β′ += generator.β
    end

    relation′ = GaussianRelation(A′, B′, a′, b′, β′)

    # delete
    mask = falses(n)
    section = Int[]

    for p in parts(diagram, :OuterPort)
        j = diagram[p, :outer_junction]

        if !mask[j]
            mask[j] = true
            push!(section, p)
        end
    end

    relation′ = marginalize(relation′, mask)

    # copy
    A[section, section] .= relation′.A
    B[section, section] .= relation′.B
    a[section] .= relation′.a
    b[section] .= relation′.b
    β = relation′.β

    for p in section
        j = diagram[p, :outer_junction]
        ports = incident(diagram, j, :outer_junction)
        B[ports, ports] .+= 3I(length(ports))
        B[ports, ports] .-= 1
    end

    relation = GaussianRelation(A, B, a, b, β)
    relation
end
