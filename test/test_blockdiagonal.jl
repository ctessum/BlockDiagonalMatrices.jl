using BlockDiagonalMatrices
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test

function get_random_sized_psdm(n)
    b1 = rand(n,n);
    Q = qr(b1).Q
    B1 = Q*Diagonal((rand(n) .+ 0.5 ))*Q'
    return (B1 + B1')/2
end

#### Testing equal block sizes

# Number of blocks
n = 100
# Make 3x3 blocks
B = BlockDiagonal([get_random_sized_psdm(3) for i = 1:n])
x = rand(3n)
X = rand(3n,3)
# Creating sparse representation
SB = sparse(B)
# Comparing against the sparse matrix
@test B*x ≈ SB*x
@test B\x ≈ SB\x
@test B*X ≈ SB*X
@test B\X ≈ SB\X
@test sparse(B) == SB
# Testing logs, determinants, and variants
@test logdet(B) ≈ logdet(SB)
@test diag(B) ≈ diag(SB)
@test det(B) ≈ det(SB)
@test tr(B) ≈ tr(SB)
# Testing indicies
@test size(B) == size(SB)
@test B[n,2n] == SB[n,2n]
@test B[2n,n] == SB[2n,n]
@test B[1,1] == SB[1,1]
@test B[end,end] == SB[end,end]

B_dense = Matrix(B)
b1 = size(B.blocks[1],1)
bn = size(B.blocks[end],1) - 1
for func in (:log, :sqrt, :sin, :tan, :cos, :sinh, :tanh)
    @eval begin
        func_dense = ($func)(B_dense)
        func_block = ($func)(B)
        @test func_dense[1:b1,1:b1] ≈ func_block.blocks[1]
        @test func_dense[end-bn:end,end-bn:end] ≈ func_block.blocks[end]
    end
end


# Make 3x3 blocks - But with StaticArrays for speedup
C = BlockDiagonal([SMatrix{3,3,eltype(block)}(block) for block in B.blocks])
S = sparse(C)
@test C*x ≈ S*x
@test C\x ≈ S\x
@test C*X ≈ S*X
@test C\X ≈ S\X
@test sparse(C) == S
# Testing logs, determinants, and variants
@test logdet(B) ≈ logdet(S)
@test diag(B) ≈ diag(S)
@test det(B) ≈ det(S)
@test tr(B) ≈ tr(S)
# Testing indicies
@test size(B) == size(S)
@test B[n,2n] == S[n,2n]
@test B[2n,n] == S[2n,n]
@test B[1,1] == S[1,1]
@test B[end,end] == S[end,end]

#### Testing blocks with different sizes (still PSD!)
# Number of blocks
n = 10
Br= BlockDiagonal([get_random_sized_psdm(i) for i = 2:n])
A = sparse(Br) # Spase representation

# Testing vector operation
xr = randn(size(Br,1))
@test A*xr ≈ Br*xr
@test A\xr ≈ Br\xr
# Testing matrix operation
Xr = randn(size(Br,1),3)
@test A*Xr ≈ Br*Xr
@test A\Xr ≈ Br\Xr

A_dense = Matrix(Br)
r1 = size(Br.blocks[1],1)
rn = size(Br.blocks[end],1) - 1
for func in (:log, :sqrt, :sin, :tan, :cos, :sinh, :tanh)
    @eval begin
        func_dense = ($func)(A_dense)
        func_block = ($func)(Br)
        @test func_dense[1:r1,1:r1] ≈ func_block.blocks[1]
        @test func_dense[end-rn:end,end-rn:end] ≈ func_block.blocks[end]
    end
end

# Testing logs, determinants, and variants
@test logdet(A) ≈ logdet(Br)
@test diag(A) ≈ diag(Br)
@test det(A) ≈ det(Br)
@test tr(A) ≈ tr(Br)

# Testing getindex
@test size(Br) == size(A)
@test Br[n,2n] == A[n,2n]
@test Br[2n,n] == A[2n,n]
@test Br[1,1] == A[1,1]
@test Br[end,end] == A[end,end]

## Testing eigenvalue decomposition
Er = eigen(Br)
E  = eigen(Matrix(A)) # SparseArrays do not support eigenvalue decomposition (Arpack does)

@test E.values ≈ sort(Er.values) # We have to sort due to the block diagonal eigenvalues coming from each block
# iperm = invperm(sortperm(Er.values))
