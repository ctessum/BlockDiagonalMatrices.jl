using BlockDiagonalMatrices
using LinearAlgebra
using SparseArrays
using StaticArrays

function get_random_sized_psdm(n)
    b1 = rand(n,n);
    Q = qr(b1).Q
    B1 = Q*Diagonal(rand(n))*Q'
    B1 = Symmetric(B1)
    return B1
end

n = 100

B = BlockDiagonal([get_random_sized_psdm(3) for i = 1:n])
C = BlockDiagonal([SMatrix{3,3,eltype(block)}(block) for block in B.blocks])

v = reshape(Vector(1:3n),3,n)
Ii = repeat(v,inner=(3,1))[:]
Jj = repeat(v,outer=3)[:]
Vv = hcat(B.blocks...)[:]
S = sparse(Ii,Jj,Vv)

x = rand(3n)
@test B*x ≈ S*x
@test B\x ≈ S\x
@test C*x ≈ S*x
@test C\x ≈ S\x
X = rand(3n,3)
@test B*X ≈ S*X
@test B\X ≈ S\X
@test C*X ≈ S*X
@test C\X ≈ S\X

#
@test sparse(B) == S
@test sparse(C) == S

# Un-equal blocks
n = 10
Br= BlockDiagonal([get_random_sized_psdm(i) for i = 2:n])
xr = randn(size(Br,1))

A = Matrix(Br)

@test A*xr ≈ Br*xr
@test A\xr ≈ Br\xr


# Extra
@test logdet(A) ≈ logdet(Br)
@test diag(A) ≈ diag(Br)
@test tr(A) ≈ tr(Br)
@test tr(A) ≈ tr(Br)

# Testing getindex
@test size(Br) == size(A)
@test Br[n,2n] == A[n,2n]
@test Br[2n,n] == A[2n,n]

# Testing eigenvalue decomposition
Er = eigen(Br)
E  = eigen(A)

@test E.values ≈ sort(Er.values)
# iperm = invperm(sortperm(Er.values))
