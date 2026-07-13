# Round-trip I/O test for Series-valued U1Nf2 configurations (U1SeriesIO.jl).
#
# Primary use case (mirrors LFTU1/main/Nf2/nf2-compute-corrs.jl): save several
# configurations, then read them back one by one with read_cnfg_info +
# read_cnfg_n / read_next_cnfg. read_ensemble is also exercised as a cross-check.

NDER = 2
N    = 1 + NDER
L    = 6
NCFG = 3

# Series-valued mass:  value 0.02, first-derivative direction = 1
am0 = Series{ComplexF64, N}(ntuple(
    i -> i == 1 ? ComplexF64(0.02) : (i == 2 ? 1.0+0im : 0.0+0im), N))

buildmodel() = U1Nf2(Float64, typeof(am0), beta = 4.0, am0 = am0,
                     iL = (L, L), BC = PeriodicBC)

# Reference configurations: arbitrary Series data in every link and every order
Random.seed!(1234)
refU = Vector{Array{Series{ComplexF64, N}, 3}}(undef, NCFG)
for c in 1:NCFG
    m = buildmodel()
    for i in eachindex(m.U)
        m.U[i] = Series{ComplexF64, N}(ntuple(k -> randn(ComplexF64), N))
    end
    refU[c] = copy(m.U)
end

fname = tempname() * ".bdio"

# --- save NCFG configurations to the same file ---
for c in 1:NCFG
    m = buildmodel()
    m.U .= refU[c]
    LFTSampling.save_cnfg(fname, m)
end

@testset "config count" begin
    @test LFTSampling.count_configs(fname) == NCFG
end

# --- primary use case: read one by one ---
@testset "header / read_cnfg_info" begin
    fb, model = read_cnfg_info(fname, U1Nf2, Series{ComplexF64, N})
    @test model.params.beta == 4.0
    @test model.params.am0  == am0
    @test model.params.iL   == (L, L)
    @test model.params.BC   == PeriodicBC
    @test eltype(model.U)   == Series{ComplexF64, N}
    close(fb)
end

@testset "auto-detect N from header (Series marker)" begin
    fb, model = read_cnfg_info(fname, U1Nf2, Series)   # no N supplied
    @test eltype(model.U) == Series{ComplexF64, N}
    @test model.params.am0 == am0
    read_next_cnfg(fb, model)
    @test model.U == refU[1]
    close(fb)
end

@testset "read configs one by one" begin
    start = 1
    fb, model = read_cnfg_info(fname, U1Nf2, Series{ComplexF64, N})
    for i in start:NCFG
        if i == start && start != 1
            LFTSampling.read_cnfg_n(fb, start, model)
        else
            read_next_cnfg(fb, model)
        end
        @test model.U == refU[i]
    end
    close(fb)
end

@testset "read_cnfg_n random access" begin
    # jump straight to configuration 2
    fb, model = read_cnfg_info(fname, U1Nf2, Series{ComplexF64, N})
    LFTSampling.read_cnfg_n(fb, 2, model)
    @test model.U == refU[2]
    close(fb)
end

# --- cross-check: read_ensemble ---
@testset "read_ensemble cross-check" begin
    ens = U1Series.read_ensemble(fname, U1Nf2, Series{ComplexF64, N})
    @test length(ens) == NCFG
    for c in 1:NCFG
        @test ens[c].U == refU[c]
    end
    ens2 = U1Series.read_ensemble(fname, U1Nf2, Series{ComplexF64, N}, 2)
    @test length(ens2) == 2
end

rm(fname, force = true)
