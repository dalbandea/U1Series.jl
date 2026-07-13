# Reproducibility check (from the bottom of main/Nf2/nf2-hmc-ad1.jl):
# with a fixed RNG seed, the Series-valued HMC reproduces the plain Float64
# molecular-dynamics trajectory at constant (0th) order.
#
# Only molecular_dynamics! is run (no accept/reject, no extra rand()). The MD
# is deterministic given the momenta + pseudofermions, which draw identical
# constant-order randomness in both models when seeded equally:
#   * momenta:        randn(Series{Float64,N}) == randn(Float64)  (1 draw each)
#   * pseudofermions: both models have PRC = Float64 -> both draw the same
#                     randn(ComplexF64, iL..., 2) array.

NDER = 1
NS   = 1 + NDER
L    = 8

# Float64 model and its Series-valued counterpart (same beta / mass / lattice)
model = U1Nf2(Float64, beta = 4.0, am0 = 0.02, iL = (L, L), BC = PeriodicBC)

am0 = Series{ComplexF64, NS}(ntuple(
    i -> i == 1 ? ComplexF64(0.02) : (i == 2 ? 1.0+0im : 0.0+0im), NS))
model_s = U1Nf2(Float64, typeof(am0), beta = 4.0, am0 = am0, iL = (L, L), BC = PeriodicBC)

# Common starting configuration: a random Float64 config, copied into the
# constant order of the Series model (zero derivative orders).
Random.seed!(2024)
LFTU1.randomize!(model)
U0 = copy(model.U)
for i in eachindex(model_s.U)
    model_s.U[i] = Series{ComplexF64, NS}(ntuple(k -> k == 1 ? U0[i] : 0.0+0im, NS))
end

smplr       = HMC(integrator = Leapfrog(1.0, 10))
samplerws   = LFTSampling.sampler(model_s, smplr)
samplerws_0 = LFTSampling.sampler(model, smplr)

seed = 12345

# --- Series run ---
Random.seed!(seed)
generate_momenta!(model_s, samplerws)
generate_pseudofermions!(model_s, samplerws)
mom_s = getindex.(samplerws.mom, 1)          # constant-order momenta snapshot
F_s   = getindex.(samplerws.F, 1)            # constant-order pseudofermion
LFTSampling.molecular_dynamics!(model_s, samplerws)

# --- Float64 run ---
Random.seed!(seed)
generate_momenta!(model, samplerws_0)
generate_pseudofermions!(model, samplerws_0)
mom_0 = copy(samplerws_0.mom)
F_0   = copy(samplerws_0.F)
LFTSampling.molecular_dynamics!(model, samplerws_0)

momdiff = maximum(abs.(mom_s .- mom_0))
Fdiff   = maximum(abs.(F_s   .- F_0))
Udiff   = maximum(abs.(getindex.(model_s.U, 1) .- model.U))

@testset "constant-order MD reproducibility" begin
    @test momdiff == 0.0            # momenta: identical randn draws
    @test Fdiff   < 1e-12           # pseudofermion (one gamm5D application)
    @test Udiff   < 1e-10           # post-MD gauge field, constant order
end
