using Test
using LFTSampling
using LFTU1
using FormalSeries
using U1Series
using Random

@testset verbose = true "U1Series tests" begin

    @testset verbose = true "Series I/O" begin
        include("seriesiotests.jl")
    end

    @testset verbose = true "Series HMC reproducibility" begin
        include("reproducibilitytests.jl")
    end

    @testset verbose = true "Gauge damping" begin
        include("gaugedampingtests.jl")
    end

end
