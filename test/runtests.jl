using SafeTestsets

# Formatting
@safetestset "Aqua testing            " begin include("test_aqua.jl")              end
@safetestset "BlockDiagonal           " begin include("test_blockdiagonal.jl")     end
@safetestset "RectangularBlockDiagonal" begin include("test_rectblockdiagonal.jl") end
