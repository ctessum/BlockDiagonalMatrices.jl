using BlockDiagonalMatrices
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test

#### Testing equal block sizes
n_blocks = 10
print_on = false
long_test = false
# block_types = ["dense", "static"]
block_types = ["dense"]
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
                B = BlockDiagonal([rand(rand(1:3),2) for i = 1:n_blocks])
            elseif block_type == "static"
                Bblocks = [rand(T,rand(1:3),2) for i = 1:n_blocks]
                B = BlockDiagonal([SMatrix{size(block)...,eltype(block)}(block) for block in Bblocks])
            end
            x = rand(S,size(B,2))
            X = rand(S,size(B,2),3)
            # Creating sparse representation
            SB = sparse(B)
            # Comparing against the sparse matrix
            @test B*x ≈ SB*x
            @test B*X ≈ SB*X
            @test sparse(B) == SB
            # Testing indicies
            @test size(B) == size(SB)
            @test B[n_blocks,div(n_blocks,2)] == SB[n_blocks,div(n_blocks,2)]
            @test B[1,1] == SB[1,1]
            @test B[end,end] == SB[end,end]

            # Test basic operations
            for test_block_type in test_block_types
                # println("$(test_block_type)")
                Ablocks = [rand(S,size(block,2),size(block,1)) for block in B.blocks]
                if block_type == "dense"
                    A = BlockDiagonal(Ablocks)
                elseif block_type == "static"
                    A = BlockDiagonal([SMatrix{size(block)...,S}(block) for block in Ablocks])
                end
                Ad = Matrix(A)
                Bd = Matrix(B)
                @test  A*B  ≈  Ad*Bd
                @test A'*B' ≈ Ad'*Bd'

                @test A*(-B) ≈ Ad*(-Bd)
                @test A'*(-B') ≈ Ad'*(-Bd')

                @test A*(2B) ≈ Ad*(2Bd)
                @test (2A)*B ≈ (2Ad)*Bd

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
