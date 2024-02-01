"""
    BlockDiagonal

A (square) block-diagonal matrix
    - Following standard notation https://en.wikipedia.org/wiki/Block_matrix)
"""
struct RectangularBlockDiagonal{T, V <: AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    cumulative_row_indices::Vector{Int}
    cumulative_col_indices::Vector{Int}
end

"""
    BlockDiagonal(blocks)

Creates a (square) block-diagonal matrix
"""
function RectangularBlockDiagonal(blocks::Vector{V}) where {T, V <: AbstractMatrix{T}}
    cumulative_row_indices = cumulative_square_indices(blocks)
    cumulative_col_indices = cumulative_square_indices(blocks,dim=2)
    return RectangularBlockDiagonal{T,V}(blocks,cumulative_row_indices,cumulative_col_indices)
end

"""
    blocks(B::RectangularBlockDiagonal)

Return the on-diagonal blocks of B.
"""
blocks(B::RectangularBlockDiagonal) = B.blocks

"""
    size(B::RectangularBlockDiagonal)

Gets the size of the BlockDiagonal matrix using the cumulative indices
"""
Base.size(B::RectangularBlockDiagonal) = (B.cumulative_row_indices[end], B.cumulative_col_indices[end])

"""
    getindex(B::RectangularBlockDiagonal{T},i::Integer,j::Integer)

Grabs indexing sorting the cumulative indicies of the matrix
 - Should scale O(log(n)), as compared to O(n) of the implementation in BlockDiagonals.jl
"""
function Base.getindex(B::RectangularBlockDiagonal{T},i::Integer,j::Integer) where {T}
    row_indicies = B.cumulative_row_indices
    col_indicies = B.cumulative_col_indices
    # Find row-block
    ls = searchsorted(row_indicies, i)
    if ls.start == ls.stop
        block_no = ls.start - 1
    else
        block_no = ls.stop
    end
    # Find-block - Return 0 if now in column block
    # if j ∈ (col_indicies[ls.stop] + 1):col_indicies[ls.start]
    if j ∈ (col_indicies[block_no] + 1):col_indicies[block_no+1]
        return blocks(B)[block_no][i - row_indicies[block_no], j - col_indicies[block_no]]
    else
        return zero(eltype(B))
    end
end

"""
    Matrix(B::RectangularBlockDiagonal{T})

Converts a BlockDiagonal matrix into dense matrix
"""
function Base.Matrix(B::RectangularBlockDiagonal{T}) where {T}
    A = zeros(T, size(B))

    @inbounds for (block_id,block) in enumerate(blocks(B))
        block_row_start = B.cumulative_row_indices[block_id] + 1
        block_row_end   = B.cumulative_row_indices[block_id + 1]
        block_col_start = B.cumulative_col_indices[block_id] + 1
        block_col_end   = B.cumulative_col_indices[block_id + 1]
        A[block_row_start:block_row_end, block_col_start:block_col_end] = block
    end
    return A
end

"""
    sparse(B::RectangularBlockDiagonal{T})

Converts a BlockDiagonal matrix into sparse matrix
"""
function SparseArrays.sparse(B::RectangularBlockDiagonal{T}) where {T}
    nnz = sum(block -> prod(size(block)), blocks(B))
    I = zeros(Int, nnz)
    J = zeros(Int, nnz)
    V = zeros(nnz)
    block_row_sizes = diff(B.cumulative_row_indices)
    block_col_sizes = diff(B.cumulative_col_indices)
    linear_indices = [0; cumsum(block_row_sizes .* block_col_sizes)]
    @floop @inbounds for (block_id,block) in enumerate(blocks(B))
        idx = (linear_indices[block_id] + 1):linear_indices[block_id + 1]
        I[idx] = repeat((B.cumulative_row_indices[block_id] + 1):B.cumulative_row_indices[block_id + 1], outer=block_col_sizes[block_id])
        J[idx] = repeat((B.cumulative_col_indices[block_id] + 1):B.cumulative_col_indices[block_id + 1], inner=block_row_sizes[block_id])
        V[idx] = block[:]
    end
    return sparse(I,J,V)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat{T}, B::RectangularBlockDiagonal{T,V}, x::AbstractVecOrMat{T}) where {T,V}
    @floop @inbounds  for (block_id, block) in enumerate(blocks(B))
        block_row_start = B.cumulative_row_indices[block_id] + 1
        block_row_end   = B.cumulative_row_indices[block_id + 1]
        block_col_start = B.cumulative_col_indices[block_id] + 1
        block_col_end   = B.cumulative_col_indices[block_id + 1]
        mul!(selectdim(y,1,block_row_start:block_row_end), block, selectdim(x,1,block_col_start:block_col_end))
    end
    return y
end
