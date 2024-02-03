"""
    BlockDiagonal

A block-diagonal matrix
"""
struct BlockDiagonal{T, V <: AbstractMatrix{T}} <: AbstractBlockDiagonal{T}
    blocks::Vector{V}
    block_row_indices::Vector{Int}
    block_col_indices::Vector{Int}
    is_block_square::Bool
end


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
    BlockDiagonal(blocks)

Creates a (square) block-diagonal matrix
"""
function BlockDiagonal(blocks::Vector{V}) where {T, V <: AbstractMatrix{T}}
    block_row_indices = cumulative_square_indices(blocks)
    block_col_indices = cumulative_square_indices(blocks,dim=2)
    is_block_square = all(block -> _is_square(block), blocks)
    return BlockDiagonal{T,V}(blocks,block_row_indices,block_col_indices,is_block_square)
end


"""
    size(B::BlockDiagonal)

Gets the size of the BlockDiagonal matrix using the cumulative indices
"""
Base.size(B::BlockDiagonal) = (B.block_row_indices[end], B.block_col_indices[end])

# Helper function. Extracts indicies and information about square blocks
_extract_block_information(B::BlockDiagonal) = B.block_row_indices, B.block_col_indices, B.is_block_square

"""
    getindex(B::BlockDiagonal{T},i::Integer,j::Integer)

Grabs indexing sorting the cumulative indicies of the matrix
 - Should scale O(log(n)), as compared to O(n) of the implementation in BlockDiagonals.jl
"""
function Base.getindex(B::BlockDiagonal{T},i::Integer,j::Integer) where {T}
    row_indicies = B.block_row_indices
    col_indicies = B.block_col_indices
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
    Matrix(B::BlockDiagonal{T})

Converts a BlockDiagonal matrix into dense matrix
"""
function Base.Matrix(B::BlockDiagonal{T}) where {T}
    A = zeros(T, size(B))
    @floop @inbounds for (block_id,block) in enumerate(blocks(B))
        block_row_start = B.block_row_indices[block_id] + 1
        block_row_end   = B.block_row_indices[block_id + 1]
        block_col_start = B.block_col_indices[block_id] + 1
        block_col_end   = B.block_col_indices[block_id + 1]
        A[block_row_start:block_row_end, block_col_start:block_col_end] = block
    end
    return A
end

"""
    sparse(B::BlockDiagonal{T})

Converts a BlockDiagonal matrix into sparse matrix
"""
function SparseArrays.sparse(B::BlockDiagonal{T}) where {T}
    nnz = sum(block -> prod(size(block)), blocks(B))
    I = zeros(Int, nnz)
    J = zeros(Int, nnz)
    V = zeros(T,nnz)
    block_row_sizes = diff(B.block_row_indices)
    block_col_sizes = diff(B.block_col_indices)
    linear_indices = [0; cumsum(block_row_sizes .* block_col_sizes)]
    @floop @inbounds for (block_id,block) in enumerate(blocks(B))
        idx = (linear_indices[block_id] + 1):linear_indices[block_id + 1]
        I[idx] = repeat((B.block_row_indices[block_id] + 1):B.block_row_indices[block_id + 1], outer=block_col_sizes[block_id])
        J[idx] = repeat((B.block_col_indices[block_id] + 1):B.block_col_indices[block_id + 1], inner=block_row_sizes[block_id])
        V[idx] = block[:]
    end
    return SparseArrays.sparse(I,J,V)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat{T}, B::BlockDiagonal{T,V}, x::AbstractVecOrMat{T}) where {T,V}
    @floop @inbounds for (block_id, block) in enumerate(blocks(B))
        block_row_start = B.block_row_indices[block_id] + 1
        block_row_end   = B.block_row_indices[block_id + 1]
        block_col_start = B.block_col_indices[block_id] + 1
        block_col_end   = B.block_col_indices[block_id + 1]
        @views mul!(y[block_row_start:block_row_end,:], block, x[block_col_start:block_col_end,:])
    end
    return y
end


LinearAlgebra.logdet(B::BlockDiagonal) = B.is_block_square ? sum(logdet, blocks(B)) : error("Not all blocks are square")
LinearAlgebra.det(B::BlockDiagonal) = B.is_block_square ? prod(det, blocks(B)) : error("Not all blocks are square")
LinearAlgebra.tr(B::BlockDiagonal) = B.is_block_square ? sum(tr, blocks(B)) : sum(diag(B))


Base.transpose(B::BlockDiagonal) = BlockDiagonal([transpose(block) for block in blocks(B)], B.block_col_indices, B.block_row_indices, B.is_block_square)
Base.adjoint(B::BlockDiagonal) = BlockDiagonal([transpose(conj(block)) for block in blocks(B)], B.block_col_indices, B.block_row_indices, B.is_block_square)


Base.:*(scalar::Number, B::BlockDiagonal) = BlockDiagonal(scalar * blocks(B), _extract_block_information(B)...)
Base.:*(B::BlockDiagonal,scalar::Number)  = BlockDiagonal(scalar * blocks(B), _extract_block_information(B)...)


Base.:-(B::BlockDiagonal) = BlockDiagonal(-blocks(B), _extract_block_information(B)...)
Base.:+(B::BlockDiagonal) = BlockDiagonal(blocks(B), _extract_block_information(B)...)


function Base.:-(B::BlockDiagonal, C::BlockDiagonal)
    if !((B.block_row_indices == C.block_row_indices) && (B.block_col_indices == C.block_col_indices))
        throw(DimensionMismatch("The block sizes of A and B are not compatible"))
    end
    return BlockDiagonal(blocks(B) - blocks(C), _extract_block_information(B)...)
end
function Base.:+(B::BlockDiagonal, C::BlockDiagonal)
    if !((B.block_row_indices == C.block_row_indices) && (B.block_col_indices == C.block_col_indices))
        throw(DimensionMismatch("The block sizes of A and B are not compatible"))
    end
    return BlockDiagonal(blocks(B) + blocks(C), _extract_block_information(B)...)
end
function Base.:*(B::BlockDiagonal,C::BlockDiagonal)
    if !(B.block_col_indices == C.block_row_indices)
        throw(DimensionMismatch("The block sizes of A and B are not compatible"))
    end
    W = [zeros(promote_type(eltype(B),eltype(C)),size(B_block,1), size(C_block,2)) for (B_block, C_block) in zip(blocks(B),blocks(C))]
    @floop @inbounds for (block_id, (B_block, C_block)) in enumerate(zip(blocks(B),blocks(C)))
        @views mul!(W[block_id], B_block, C_block)
    end
    is_block_square = all(block -> _is_square(block), W)
    return BlockDiagonal(W,B.block_row_indices, C.block_col_indices, is_block_square)
end


function LinearAlgebra.:\(B::BlockDiagonal, x::AbstractVecOrMat)
    if !B.is_block_square
        throw(DimensionMismatch("Not all blocks are square. Use a sparse format instead i.e. sparse(B)."))
    end
    y = zeros(promote_type(eltype(B),eltype(x)),size(x))
    @floop @inbounds for (block_id,block) in enumerate(blocks(B))
        block_start = B.block_row_indices[block_id] + 1
        block_end   = B.block_row_indices[block_id + 1]
        # F = cholesky(block)
        # F = factorize(block)
        # Some problems with ldiv! when using sparse blocks
        # @views ldiv!(y[block_start:block_end,:], F, x[block_start:block_end,:])
        @views y[block_start:block_end,:] = block \ x[block_start:block_end,:]
    end
    return y
end

function LinearAlgebra.:\(B::BlockDiagonal, C::BlockDiagonal)
    if !(B.block_col_indices == C.block_row_indices)
        throw(DimensionMismatch("The block sizes of A and B are not compatible"))
    end
    if !B.is_block_square
        throw(DimensionMismatch("Not all blocks are square. Use a sparse format instead i.e. sparse(B)."))
    end
    W = [zeros(promote_type(eltype(B),eltype(C)),size(B_block,1), size(C_block,2)) for (B_block, C_block) in zip(blocks(B),blocks(C))]
    @floop @inbounds for (block_id, (B_block, C_block)) in enumerate(zip(blocks(B),blocks(C)))
        L = factorize(B_block)
        # ldiv! have problems with some sparse arrays
        @views W[block_id] = L \ C_block
        # @views ldiv!(W[block_id], L , C_block)
    end
    is_block_square = all(block -> _is_square(block), W)
    return BlockDiagonal(W,B.block_row_indices, C.block_col_indices, is_block_square)
end

## Functions
# Idea to include matrix functions?
# for func in (:log, :sqrt, :sin, :tan, :cos, :sinh, :tanh)
#     @eval begin
#         function (Base.$func)(B::BlockDiagonal)
#             if !B.is_block_square
#                 error("Matrix not block square")
#             end
#             return BlockDiagonal([($func)(block) for block in blocks(B)], _extract_block_information(B)...)
#         end
#     end
# end
