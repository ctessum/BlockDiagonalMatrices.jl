module BlockDiagonalMatrices

using FLoops
using LinearAlgebra
using SparseArrays

include("blockdiagonal.jl")
include("rectangularblockdiagonal.jl")

export BlockDiagonal
export RectangularBlockDiagonal

end
