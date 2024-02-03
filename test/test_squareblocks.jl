using BlockDiagonalMatrices
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test

#### Testing equal block sizes
n_blocks = 10
print_on = false
long_test = false
# block_types = ["dense", "sparse", "static"]
block_types = ["dense", "mixed"]
for T in [Float32,Float64,ComplexF32, ComplexF64]
    for block_type in block_types
        @testset "$(block_type) | $(T)" begin
            if long_test
                test_block_eltypes = [Float32,Float64,ComplexF32, ComplexF64]
                test_block_types = block_types
            else
                test_block_eltypes = [T]
                test_block_types = [block_type]
            end
        for S in test_block_eltypes
            if print_on
                println("$(T) | $(S) | $(block_type)" ) # For debugging test
            end
            if block_type == "dense"
                B = BlockDiagonal([rand(T,i,i) + I for i = 1:n_blocks])
            elseif block_type == "sparse"
                B = BlockDiagonal([sprand(T,i,i,0.4) + I for i = 1:n_blocks])
            elseif block_type == "static"
                B = BlockDiagonal([SMatrix{3,3,T}(rand(T,3,3)) for i = 1:n_blocks])
            elseif block_type == "mixed"
                B1blocks = [rand(T,i,i) + I for i = 1:div(n_blocks,2)]
                B2blocks = [sprand(T,i,i,0.4) + I for i = div(n_blocks,2)+1:n_blocks]
                B = BlockDiagonal(vcat(B1blocks,B2blocks))
            end
            x = rand(S,size(B,2))
            X = rand(S,size(B,2),3)
            # Creating sparse representation
            SB = sparse(B)
            # Comparing against the sparse matrix
            @test B*x ≈ SB*x
            @test B\x ≈ SB\x    rtol = sqrt(eps(Float32))
            @test B*X ≈ SB*X
            @test B\X ≈ SB\X    rtol = sqrt(eps(Float32))
            @test sparse(B) == SB
            # Testing logs, determinants, and variants
            # @test logdet(B) ≈ logdet(SB)
            @test diag(B) ≈ diag(SB)
            @test det(B) ≈ det(SB)      rtol = sqrt(eps(Float32))
            @test tr(B) ≈ tr(SB)
            # Testing indicies
            @test size(B) == size(SB)
            @test B[n_blocks,2n_blocks] == SB[n_blocks,2n_blocks]
            @test B[2n_blocks,n_blocks] == SB[2n_blocks,n_blocks]
            @test B[1,1] == SB[1,1]
            @test B[end,end] == SB[end,end]

            @test transpose(B) == transpose(SB)
            @test adjoint(B) == adjoint(SB)

            # Test basic operations
            for test_block_type in test_block_types
                # println("$(test_block_type)")
                if block_type == "dense"
                    A = BlockDiagonal([rand(S,i,i) + I for i = 1:n_blocks])
                elseif block_type == "sparse"
                    A = BlockDiagonal([sprand(S,i,i,0.4)+I for i = 1:n_blocks])
                elseif block_type == "static"
                    A = BlockDiagonal([SMatrix{3,3,S}(rand(S,3,3)) for i = 1:n_blocks])
                elseif block_type == "mixed"
                    A1blocks = [sprand(T,i,i,0.4)+I for i = 1:div(n_blocks,2)]
                    A2blocks = [rand(T,i,i) + I for i = div(n_blocks,2)+1:n_blocks]
                    A = BlockDiagonal(vcat(A1blocks,A2blocks))
                end
                Ad = Matrix(A)
                Bd = Matrix(B)
                @test  A*B  ≈  Ad*Bd
                @test A'*B' ≈ Ad'*Bd'
                @test  A*B' ≈  Ad*Bd'
                @test A'*B  ≈ Ad'*Bd

                @test A\B ≈ Ad\Bd
                @test A'\B ≈ Ad'\Bd
                @test A\B' ≈ Ad\Bd'

                @test A*(-B) ≈ Ad*(-Bd)
                @test A'*(-B') ≈ Ad'*(-Bd')
                @test -A*B  ≈ -Ad*Bd
                @test -A'*B ≈ -Ad'*Bd

                @test A*(2B) ≈ Ad*(2Bd)

                @test  A + B ≈  Ad + Bd
                @test  A - B ≈  Ad - Bd
                @test -A - B ≈ -Ad - Bd
                @test -A + B ≈ -Ad + Bd

                @test  A + 2B ≈  Ad + 2Bd
                @test  A - 2B ≈  Ad - 2Bd
                @test -A - 2B ≈ -Ad - 2Bd
                @test -A + 2B ≈ -Ad + 2Bd

                @test  A + B*2 ≈  Ad + Bd*2
                @test  A - B*2 ≈  Ad - Bd*2
                @test -A - B*2 ≈ -Ad - Bd*2
                @test -A + B*2 ≈ -Ad + Bd*2
            end


            #### TODO: Revisit tests
            # if T <: Real && block_type == "dense"
            #     for func in (:log, :sqrt, :sin, :tan, :cos, :sinh, :tanh)
            #         @eval begin
            #             b1 = size(B.blocks[1],1)
            #             bn = size(B.blocks[end],1) - 1
            #             B_dense = Matrix(B)
            #             func_block = ($func)(B)
            #             func_dense = ($func)(B_dense)
            #             @test func_dense[1:b1,1:b1] ≈ func_block.blocks[1]
            #             @test func_dense[end-bn:end,end-bn:end] ≈ func_block.blocks[end]
            #         end
            #     end
            # end

            ### Testing eigenvalue decomposition - Problem
            # Have to revisit how to test this. Only certain real eigen values for PSD?
            # Er = eigen(B)
            # E  = eigen(Matrix(B)) # SparseArrays do not support eigenvalue decomposition (Arpack does)
            # if T <: Real # Can not sort complex numbers
            #     @test E.values ≈ sort(Er.values) # We have to sort due to the block diagonal eigenvalues coming from each block
            # end
            # iperm = invperm(sortperm(Er.values))
            end
        end
    end
end
