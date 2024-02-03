using SafeTestsets

# Formatting
@safetestset "Aqua testing      " begin include("test_aqua.jl")              end
@safetestset "Square blocks     " begin include("test_squareblocks.jl")      end
@safetestset "Rectangular blocks" begin include("test_rectangularblocks.jl") end
