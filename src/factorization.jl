
"""
The result of a LU factorization of a block diagonal matrix.
"""
struct BlockDiagonalLU{T} <: AbstractBlockDiagonal{T}
    blocks::Vector{T}
end

function LinearAlgebra.issuccess(F::BlockDiagonalLU; kwargs...)
    for b in blocks(F)
        if !LinearAlgebra.issuccess(b; kwargs...)
            return false
        end
    end
    return true
end

function ArrayInterface.lu_instance(A::AbstractBlockDiagonal)
    return BlockDiagonalLU([ArrayInterface.lu_instance(b) for b in blocks(A)])
end

function LinearAlgebra.lu!(B::AbstractBlockDiagonal, args...; kwargs...)
    BlockDiagonalLU([lu!(blk, args...; kwargs...) for blk in blocks(B)])
end

function LinearAlgebra.lu(B::AbstractBlockDiagonal, args...; kwargs...)
    BlockDiagonalLU([lu(blk, args...; kwargs...) for blk in blocks(B)])
end

function LinearAlgebra.ldiv!(x::AbstractVecOrMat, A::BlockDiagonalLU, b::AbstractVecOrMat; kwargs...)
    row_i = 1
    @assert size(x) == size(b) "dimensions of x and b must match"
    @assert mapreduce(a -> size(a, 1), +, blocks(A)) == size(b, 1) "number of rows must match"
    for block in blocks(A)
        nrow = size(block, 1)
        _x = view(x, row_i:(row_i + nrow - 1), :)
        _b = view(b, row_i:(row_i + nrow - 1), :)
        ldiv!(_x, block, _b; kwargs...)
        row_i += nrow
    end
    x
end
