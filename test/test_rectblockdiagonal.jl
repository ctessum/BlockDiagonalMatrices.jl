using BlockDiagonalMatrices
using SparseArrays
n = 100

B = RectangularBlockDiagonal([rand(rand(1:3),rand(1:3)) for i = 1:n])
A = Matrix(B)
S = sparse(B)
N = size(B,2)

# For now only multiplication is implemented
x = rand(N)
@test B*x ≈ A*x
@test B*x ≈ S*x
X = rand(N,3)
@test B*X ≈ A*X
@test B*X ≈ S*X

# Testing getindex
M = div(N,2)
@test size(B) == size(A)
@test size(B) == size(S)
@test B[M,M] == A[M,M]
@test B[M,M] == S[M,N]
@test B[1,3] == A[1,3]
@test B[1,3] == S[1,3]
# Last block, has to be zero up and down
I = B.cumulative_row_indices[n]
J = B.cumulative_col_indices[n]
@test B[I + 1,J] == 0
@test B[I,J + 1] == 0
