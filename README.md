# BlockDiagonalMatrices

[![Build Status](https://github.com/mipals/BlockDiagonalMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mipals/BlockDiagonalMatrices.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Introduction
A block diagonal matrix is a [square matrix](https://en.wikipedia.org/wiki/Block_matrix#Block_diagonal_matrices) given by

$$
\mathbf{B} = 
\begin{bmatrix}
    \mathbf{B}_1 & \mathbf{0} & \dots & \mathbf{0}\\
    \mathbf{0}   & \mathbf{B}_2 & \dots & \mathbf{0}\\
    \vdots       & \vdots & \ddots & \vdots\\
    \mathbf{0}   & \mathbf{0} & \dots & \mathbf{B}_n\\
\end{bmatrix} \in \mathbf{R}^{n\times n}
$$

where each block, given as $\mathbf{B}_i$, is also square. In simple terms this matrix structure represents a system of linear equations that are block separated. A result of the separation is that the system is only solvable (invertible) when each separated block is solvable. Unsurprisingly the block diagonal matrix system can in this case be solved by solving each of the block systems separately

$$
\mathbf{B}^{-1} = 
\begin{bmatrix}
    \mathbf{B}_1^{-1} & \mathbf{0} & \dots & \mathbf{0}\\
    \mathbf{0}   & \mathbf{B}_2^{-1} & \dots & \mathbf{0}\\
    \vdots       & \vdots & \ddots & \vdots\\
    \mathbf{0}   & \mathbf{0} & \dots & \mathbf{B}_n^{-1}\\
\end{bmatrix} \in \mathbf{R}^{n\times n}
$$

In addition, the structure also result in easy computation of traces and determinants as

$$
\text{tr}\left(\mathbf{B}\right) = \sum_{i=1}^n\text{tr}\left(\mathbf{B}_i\right)\\
$$

$$
\text{det}\left(\mathbf{B}\right) = \prod_{i=1}^n\text{det}\left(\mathbf{B}_i\right)\\
$$

$$
\text{logdet}\left(\mathbf{B}\right) = \sum_{i=1}^n\text{log}\left(\text{det}\left(\mathbf{B}_i\right)\right)
$$

Furthermore factorizations, such as the eigenvalue decomposition, can be computed separated as

$$
\mathbf{B} = 
\begin{bmatrix}
    \mathbf{Q}_1 & \mathbf{0} & \dots & \mathbf{0}\\
    \mathbf{0}   & \mathbf{Q}_2 & \dots & \mathbf{0}\\
    \vdots       & \vdots & \ddots & \vdots\\
    \mathbf{0}   & \mathbf{0} & \dots & \mathbf{Q}_n\\
\end{bmatrix}
\begin{bmatrix}
    \mathbf{\Lambda}_1 & \mathbf{0} & \dots & \mathbf{0}\\
    \mathbf{0}   & \mathbf{\Lambda}_2 & \dots & \mathbf{0}\\
    \vdots       & \vdots & \ddots & \vdots\\
    \mathbf{0}   & \mathbf{0} & \dots & \mathbf{\Lambda}_n\\
\end{bmatrix}
\begin{bmatrix}
    \mathbf{Q}_1 & \mathbf{0} & \dots & \mathbf{0}\\
    \mathbf{0}   & \mathbf{Q}_2 & \dots & \mathbf{0}\\
    \vdots       & \vdots & \ddots & \vdots\\
    \mathbf{0}   & \mathbf{0} & \dots & \mathbf{Q}_n\\
\end{bmatrix}^{-1}
$$

The package also includes some operations for rectangular block matrices.

## Examples
### ```BlockDiagonal```
First we define a block diagonal matrix with $n$ blocks of size $3\times3$.
```julia
n = 100 # Number fo blocks
B = BlockDiagonal([rand(3,3) for i = 1:n]) 
x = randn(3n)
y = B*x
norm(x - B\y)
```
The block diagonal matrices can be converted to sparse of dense arrays as follows

```julia
S = sparse(B)
M = Matrix(B)
```

Traces, determinants and log-determinant can be computed efficiently as
```julia
tr(B)
det(B)
logdet(B)
```

### ```RectangularBlockDiagonal```
A rectangular block diagonal matrix can be created similarly as
```julia
# Number of blocks
n = 100 
# Creating n
B = RectangularBlockDiagonal([rand(rand(1:3),rand(1:3)) for i = 1:n])
```
Currently there is only simple operations available for this matrix types.

