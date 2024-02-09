"""
    BlockDiagonal

A block-diagonal matrix.
Contains the blocks (`blocks`), their global position in the matrix (`block_row_indices`, `block_col_indices`), and information about if all blocks are square (`is_block_square`).
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

Creates a block-diagonal matrix given a vector of the `blocks`.
"""
function BlockDiagonal(blocks::Vector{V}) where {T, V <: AbstractMatrix{T}}
    block_row_indices = cumulative_square_indices(blocks)
    block_col_indices = cumulative_square_indices(blocks,dim=2)
    is_block_square = all(block -> _is_square(block), blocks)
    return BlockDiagonal{T,V}(blocks,block_row_indices,block_col_indices,is_block_square)
end


"""
    size(B::BlockDiagonal)

Gets the size of the BlockDiagonal matrix using the cumulative indices.
"""
Base.size(B::BlockDiagonal) = (B.block_row_indices[end], B.block_col_indices[end])

# Helper function. Extracts indicies and information about square blocks
_extract_block_information(B::BlockDiagonal) = B.block_row_indices, B.block_col_indices, B.is_block_square
_extract_transpose_block_information(B::BlockDiagonal) = B.block_col_indices, B.block_row_indices, B.is_block_square

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
    # Find corresponding column-block - Return 0 if `j` is not contained in the column block
    if j ∈ (col_indicies[block_no] + 1):col_indicies[block_no+1]
        return blocks(B)[block_no][i - row_indicies[block_no], j - col_indicies[block_no]]
    else
        return zero(T)
    end
end

"""
    Matrix(B::BlockDiagonal{T})

Converts a BlockDiagonal matrix into dense matrix.
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

Converts a BlockDiagonal matrix into sparse matrix.
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
    return sparse(I,J,V)
end

# Does not seem to be faster than [inv(block) for block in blocks(B)]
# function _create_inverse_blocks(blocks)
#     W = similar(blocks)
#     @floop @inbounds for (block_id,block) in enumerate(blocks)
#         W[block_id] = inv(Matrix(block))
#     end
#     return W
# end
"""
    inv(B::BlockDiagonal)

Inverse of a block diagonal matrix by inversion of each block.
Returns a `BlockDiagonal` of same size of `B`.
"""
function Base.inv(B::BlockDiagonal)
    return BlockDiagonal([inv(Matrix(block)) for block in blocks(B)], _extract_block_information(B)...)
end
"""
    inv_sparse(B::BlockDiagonal)

Inverse of a block diagonal matrix by inversion of each block.
Returns a sparse matrix of same size of `B`.
"""
function inv_sparse(B::BlockDiagonal{T}) where {T}
    return sparse(inv(B))
end
"""
    \\(B::BlockDiagonal, x::AbstractSparseArray)

Solving B\\A by the means of direct inversion of the blocks `B` casted into a sparse format and then performing the sparse product `B^{-1}x`.
Returns a sparse matrix.
"""
function LinearAlgebra.:\(B::BlockDiagonal, x::AbstractSparseArray)
    # NB this based on direct inversion of blocks - Mqaybe not stable?
    return inv_sparse(B)*x
end
"""
    *(B::BlockDiagonal, x::AbstractSparseArray)

Computing B*A by casting `B` into a sparse format and then performing the sparse product B*x.
Returns a sparse matrix.
"""
function LinearAlgebra.:*(B::BlockDiagonal, x::AbstractSparseArray)
    return sparse(B)*x
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


function LinearAlgebra.:\(B::BlockDiagonal, x::AbstractVecOrMat)
    if !B.is_block_square
        throw(DimensionMismatch("Not all blocks are square. Use a sparse format instead i.e. sparse(B)."))
    end
    y = similar(x,promote_type(eltype(B),eltype(x)),size(x))
    @floop @inbounds for (block_id,block) in enumerate(blocks(B))
        block_start = B.block_row_indices[block_id] + 1
        block_end   = B.block_row_indices[block_id + 1]
        # Some problems with ldiv! when using sparse blocks
        # F = cholesky(block)
        # F = factorize(block)
        # @views ldiv!(y[block_start:block_end,:], F, x[block_start:block_end,:])
        @views y[block_start:block_end,:] = block \ x[block_start:block_end,:]
    end
    return y
end

# For square blocks some efficient implementations exist
LinearAlgebra.logdet(B::BlockDiagonal) = B.is_block_square ? sum(logdet, blocks(B)) : error("Not all blocks are square")
LinearAlgebra.det(B::BlockDiagonal) = B.is_block_square ? prod(det, blocks(B)) : error("Not all blocks are square")
function LinearAlgebra.tr(B::BlockDiagonal)
    # Check if block is square. If not error. If all blocks are square sum over traces. Otherwise just sum diagonal entries
    if _is_square(B)
        return B.is_block_square ? sum(tr, blocks(B)) : sum(diag(B))
    else
        throw(DimensionMismatch("matrix is not square: dimensions are $(size(B))"))
    end
end

# Defining transposes and adjoints by applythem it to all blocks and switching the block indices
Base.transpose(B::BlockDiagonal) = BlockDiagonal([transpose(block) for block in blocks(B)], _extract_transpose_block_information(B)...)
Base.adjoint(B::BlockDiagonal) = BlockDiagonal([transpose(conj(block)) for block in blocks(B)], _extract_transpose_block_information(B)...)

# Defining some arithmetic operations of scalars
Base.:*(scalar::Number, B::BlockDiagonal) = BlockDiagonal(scalar * blocks(B), _extract_block_information(B)...)
Base.:*(B::BlockDiagonal,scalar::Number)  = BlockDiagonal(scalar * blocks(B), _extract_block_information(B)...)
Base.:-(B::BlockDiagonal) = BlockDiagonal(-blocks(B), _extract_block_information(B)...)
Base.:+(B::BlockDiagonal) = BlockDiagonal(blocks(B), _extract_block_information(B)...)

# Defining arithmetic operations with other `BlockDiagonal` matrices
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
function LinearAlgebra.:\(B::BlockDiagonal, C::BlockDiagonal)
    if !B.is_block_square
        throw(DimensionMismatch("Not all blocks are square. Use a sparse format instead i.e. sparse(B)."))
    end
    if !(B.block_col_indices == C.block_row_indices)
        throw(DimensionMismatch("The block sizes of A and B are not compatible"))
    end
    W = [zeros(promote_type(eltype(B),eltype(C)),size(B_block,1), size(C_block,2)) for (B_block, C_block) in zip(blocks(B),blocks(C))]
    @floop @inbounds for (block_id, (B_block, C_block)) in enumerate(zip(blocks(B),blocks(C)))
        L = factorize(B_block)
        # ldiv! have problems with some sparse arrays
        # @views ldiv!(W[block_id], L , C_block)
        W[block_id] .= L \ C_block
    end
    is_block_square = all(block -> _is_square(block), W)
    return BlockDiagonal(W,B.block_row_indices, C.block_col_indices, is_block_square)
end

## Functions
# Should we include matrix functions? (I do not really know if this is best approach or not. Seems slow.)
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


### First attempts at making solove with sparse rhs faster
# they're faster - But stil slower than casting the block-diagonal to sparse first;
# function LinearAlgebra.:*(B::BlockDiagonal, x::AbstractSparseArray)
#     Ii = zeros(Int64, 0)
#     Jj = zeros(Int64, 0)
#     Vv = zeros(promote_type(eltype(B),eltype(x)),0)
#     # @floop @inbounds for (block_id, block) in enumerate(blocks(B))
#     @inbounds for (block_id, block) in enumerate(B.blocks)
#         block_idx = B.block_row_indices[block_id] + 1:B.block_row_indices[block_id + 1]
#         xview = @view x[block_idx,:]
#         idx = unique(sort!(vcat([xview[i,:].nzind for i in 1:size(xview,1)]...))) # <-- Maybe make a structure for `x` where you save the non-zero block structure outside?
#         v = block*xview[:,idx]
#         # Probably not good when running @floops
#         append!(Ii,repeat(block_idx,length(idx)))
#         append!(Jj,repeat(idx,inner=length(block_idx)))
#         append!(Vv,v[:])
#     end
#     return sparse(Ii,Jj,Vv,size(B,1),size(x,1))
# end
# function LinearAlgebra.:\(B::BlockDiagonal, x::AbstractSparseArray)
#     Ii = zeros(Int64, 0)
#     Jj = zeros(Int64, 0)
#     Vv = zeros(promote_type(eltype(B),eltype(x)),0)
#     # Floop not possible due to append!?
#     # @floop @inbounds for (block_id, block) in enumerate(blocks(B))
#     @inbounds for (block_id, block) in enumerate(B.blocks)
#         block_idx = B.block_row_indices[block_id] + 1:B.block_row_indices[block_id + 1]
#         xview = @view x[block_idx,:]
#         idx = unique(sort!(vcat([xview[i,:].nzind for i in 1:size(xview,1)]...)))
#         v = block\Matrix(xview[:,idx])
#         # Probably not good when running @floops
#         append!(Ii,repeat(block_idx,length(idx)))
#         append!(Jj,repeat(idx,inner=length(block_idx)))
#         append!(Vv,v[:])
#     end
#     return sparse(Ii,Jj,Vv,size(B,1),size(x,1))
# end
