using LinearAlgebra
using SparseArrays
using StaticArrays
using FLoops

function get_random_sized_psdm(n)
    b1 = rand(n,n);
    Q = qr(b1).Q
    B1 = Q*Diagonal(rand(n))*Q'
    B1 = Symmetric(B1)
    return B1
end

n = 10000

b1 = rand(3,3);
Q = qr(b1).Q
B1 = Q*Diagonal(rand(3))*Q'
B1 = Symmetric(B1)

B = BlockDiagonal([B1 for i = 1:n])

v = reshape(Vector(1:3n),3,n)
Ii = repeat(v,inner=(3,1))[:]
Jj = repeat(v,outer=3)[:]
Vv = hcat(B.blocks...)[:]
S = sparse(Ii,Jj,Vv)

x = rand(3n)


D = BlockDiagonal([SMatrix{3,3,eltype(block)}(block) for block in B.blocks]; block_solver=solve_psd3x3)
