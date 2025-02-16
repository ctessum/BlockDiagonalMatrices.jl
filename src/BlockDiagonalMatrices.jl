module BlockDiagonalMatrices

using ArrayInterface
using FLoops
using LinearAlgebra
using SparseArrays

abstract type AbstractBlockDiagonal{T} <: AbstractMatrix{T} end
blocks(B::AbstractBlockDiagonal) = B.blocks

_is_square(A::AbstractMatrix) = size(A, 1) == size(A, 2)

include("blockdiagonal.jl")
include("factorization.jl")

export BlockDiagonal

end
