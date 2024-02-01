"""
    BlockDiagonal

A (square) block-diagonal matrix
    - Following standard notation https://en.wikipedia.org/wiki/Block_matrix)
"""
struct BlockDiagonal{T, V <: AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    cumulative_indices::Vector{Int}
end

"""
    BlockDiagonal(blocks)

Creates a (square) block-diagonal matrix
"""
function BlockDiagonal(blocks::Vector{V}) where {T, V <: AbstractMatrix{T}}
    cumulative_indices = cumulative_square_indices(blocks)
    return BlockDiagonal{T,V}(blocks,cumulative_indices)
end

"""
    blocks(B::BlockDiagonal)

Return the on-diagonal blocks of B.
"""
blocks(B::BlockDiagonal) = B.blocks

"""
    size(B::BlockDiagonal)

Gets the size of the BlockDiagonal matrix using the cumulative indices
"""
Base.size(B::BlockDiagonal) = (B.cumulative_indices[end], B.cumulative_indices[end])

"""
    cumulative_square_indices(V;dim=1)

Computes the comulative indicies of the blocks in `V` in the `dim` direction.
The vector start at zero, so the block have indicies from (index[i] + 1):index[i + 1]
"""
function cumulative_square_indices(V;dim=1)
    block_dim_size = size.(V, dim)
    return [0; cumsum(block_dim_size)]
end

"""
    getindex(B::BlockDiagonal{T},i::Integer,j::Integer)

Grabs indexing sorting the cumulative indicies of the matrix
 - Should scale O(log(n)), as compared to O(n) of the implementation in BlockDiagonals.jl
"""
function Base.getindex(B::BlockDiagonal{T},i::Integer,j::Integer) where {T}
    indicies = B.cumulative_indices
    # Find row-block
    ls = searchsorted(indicies, i)
    if ls.start == ls.stop
        block_no = ls.start - 1
    else
        block_no = ls.stop
    end
    # Find-block - Return 0 if now in column block
    if j ∈ (indicies[block_no] + 1):indicies[block_no+1]
        return blocks(B)[block_no][i - indicies[block_no], j - indicies[block_no]]
    else
        return zero(T)
    end
end

"""
    Matrix(B::BlockDiagonal{T})

Converts a BlockDiagonal matrix into dense matrix
"""
function Base.Matrix(B::BlockDiagonal{T}) where {T}
    A = zeros(T, size(B))

    @floop @inbounds for (block_id,block) in enumerate(blocks(B))
        block_start = B.cumulative_indices[block_id] + 1
        block_end   = B.cumulative_indices[block_id + 1]
        A[block_start:block_end, block_start:block_end] .= block
    end
    return A
end

"""
    sparse(B::BlockDiagonal{T})

Converts a BlockDiagonal matrix into sparse matrix
"""
function SparseArrays.sparse(B::BlockDiagonal{T}) where {T}
    nnz = sum(diff(B.cumulative_indices).^2)
    I = zeros(Int, nnz)
    J = zeros(Int, nnz)
    V = zeros(nnz)
    block_sizes = diff(B.cumulative_indices)
    indices = [0; cumsum(block_sizes.^2)]
    @floop @inbounds for (block_id,block) in enumerate(blocks(B))
        idx       = (indices[block_id] + 1):indices[block_id + 1]
        block_idx = repeat((B.cumulative_indices[block_id] + 1):B.cumulative_indices[block_id + 1], inner=(1,block_sizes[block_id]))
        I[idx] = block_idx[:]
        J[idx] = block_idx'[:]
        V[idx] = block[:]
    end
    return sparse(I,J,V)
end

# Overloading traces, determinants and diagonal blocks
LinearAlgebra.logdet(B::BlockDiagonal) = sum(logdet, blocks(B))
LinearAlgebra.det(B::BlockDiagonal) = prod(det, blocks(B))
LinearAlgebra.tr(B::BlockDiagonal) = sum(tr, blocks(B))

_iscompatible((A, B)) = size(A, 2) == size(B, 1)
function check_dim_mul(A, B)
    _iscompatible((A, B)) ||
        throw(DimensionMismatch("second dimension of left factor, $(size(A, 2)), " *
            "does not match first dimension of right factor, $(size(B, 1))"))
    return nothing
end


function LinearAlgebra.mul!(y::AbstractVecOrMat{T}, B::BlockDiagonal{T,V}, x::AbstractVecOrMat{T}) where {T,V}
    @floop @inbounds  for (block_id, block) in enumerate(blocks(B))
        block_start = B.cumulative_indices[block_id] + 1
        block_end   = B.cumulative_indices[block_id + 1]
        mul!(selectdim(y,1,block_start:block_end), block, selectdim(x,1,block_start:block_end))
    end
    return y
end

function LinearAlgebra.:\(B::BlockDiagonal, x::AbstractVecOrMat)
    y = similar(x)
    @floop @inbounds for (block_id,block) in enumerate(blocks(B))
        block_start = B.cumulative_indices[block_id] + 1
        block_end   = B.cumulative_indices[block_id + 1]
        y[block_start:block_end,:] = block\selectdim(x,1,block_start:block_end)
    end
    return y
end

"""
    eigen(B::BlockDiagonal)

Computes the eigen decomposition of `B` by using that it is equal to a block matrix where
each block is the eigen decomposition of the corresponding block of `B`.

Note that the eigen vectors are not ordered w.r.t to the eigenvalues.
"""
function LinearAlgebra.eigen(B::BlockDiagonal)
    vectors = [zeros(eltype(B),size(block)) for block in blocks(B)]
    # vectors = Vector{typeof(zeros(eltype(B),size(block)))}()
    values = zeros(eltype(B),B.cumulative_indices[end])
    @floop @inbounds for (block_id, block) in enumerate(blocks(B))
        E = eigen(block)
        vectors[block_id] .= E.vectors
        # push!(vectors,E.vectors)
        block_start = B.cumulative_indices[block_id] + 1
        block_end   = B.cumulative_indices[block_id + 1]
        values[block_start:block_end] = E.values
    end
    return Eigen(values,BlockDiagonal(vectors))
end
