struct GaussianMap{G <: GaussianRelation, M <: AbstractMatrix}
    relation::G
    matrix::M
end


function disintegrate(relation::GaussianRelation, indices₁::AbstractVector, indices₂::AbstractVector)
    i₁ = indices₁
    i₂ = indices₂

    A₁₁ = relation.A[i₁, i₂]
    A₁₂ = relation.A[i₁, i₂]
    A₂₂ = relation.A[i₂, i₂]
    B₁₁ = relation.B[i₁, i₁]
    B₁₂ = relation.B[i₁, i₂]
    B₂₂ = relation.B[i₂, i₂]
    a₁ = relation.a[i₁]
    a₂ = relation.a[i₂]
    b₁ = relation.b[i₁]
    b₂ = relation.b[i₂]
    β = relation.β

    saddlepoint = SaddlePoint(A₁₁, B₁₁)
    L = solve(saddlepoint, A₁₂, B₁₂)    
    l = solve(saddlepoint, a₁,  b₁)

    marginal = GaussianRelation(
        A₂₂ + L' * A₁₁ * L - A₁₂' * L - L' * A₁₂, 
        B₂₂ - L' * B₁₁ * L,
        a₂  + L' * A₁₁ * l - A₁₂' * l - L' * a₁,
        b₂  - L' * B₁₁ * l,
        β   - l' * B₁₁ * l)

    conditional = GaussianMap(
        GaussianRelation(A₁₁, B₁₁, a₁, b₁, l' * b),
        -L)

    marginal, conditional
end


#=
function marginalize(relation::GaussianRelation, indices::AbstractVector)
    m = length(relation)
    n = length(indices)
    indices₁ = Vector{Int}(undef, m - n)
    indices₂ = indices
    mask = falses(m)
    mask[indices] .= true    

    i = 1

    for j in 1:m
        if !mask[j]
            indices₁[i] = j
            i += 1
        end
    end

    marginal, _ = disintegrate(relation, indices₁, indices₂)
    marginal
end


function WiringDiagramAlgebras.oapply(
    diagram::UndirectedWiringDiagram,   
    generators::AbstractVector{<:GaussianRelation})

    n = nparts(diagram, :Junction)
    m = nparts(diagram, :OuterPort)
    relation = GaussianRelation(n)
    result = GaussianRelation(m)

    # merge and create
    for b in parts(diagram, :Box)
        ports = incident(diagram b, :box)
        junctions = diagram[ports, :junction]
        generator = generators[b]

        relation.A[junctions, junctions] .+= generator.A
        relation.B[junctions, junctions] .+= generator.B
        relation.a[junctions] .+= generator.a
        relation.b[junctions] .+= generator.b
        relation.β .+= generator.β
    end

    # delete
    epi, mono = epi_mono(diagram[:, :outer_junction])
    relation = marginalize(relation, mono)

    # copy
    l = length(mono)
    counts = Vector{Int}(undef, l)

    for i in 1:l
        ports = incident(diagram, mono[i], :outer_junction)
        counts[i] = length(ports)
    end

    result.A .= relation.A[epi, epi] ./ outer(counts[epi], counts[epi])
    result.B .= relation.B[epi, epi] ./ outer(counts[epi], counts[epi])
                                     .+ 3I
                                     .- I(l)[f, f]
    result.a .= relation.a[epi]
    result.b .= relation.b[epi]
    result.β .= relation.β

    result
end
=#
