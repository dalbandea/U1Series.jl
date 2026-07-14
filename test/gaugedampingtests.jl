# Gauge-damping regression tests (src/U1SeriesGaugeDamping.jl).
#
# The SMD/MD is chaotic and the KernelAbstractions kernels are not bit-reproducible
# across runs, so U^(0) *trajectories* cannot be compared. We therefore test the
# deterministic *properties* of gauge_damping! -- it is an exact gauge symmetry
# (leaves the action at every order, U^(0), and unitarity invariant) -- plus the
# statistical fact that it bounds ||A^(1)|| where plain SMD lets it diverge.

# FormalSeries: the accept/reject is non-differentiable and is skipped (as in the
# main files); the SMD sample! calls this, so it must exist.
import LFTSampling: metropolis_accept_reject!
LFTSampling.metropolis_accept_reject!(l::L, c::L, dS::FormalSeries.Series) where {L <: LFTU1.U1} =
    nothing

# --------------------------------------------------------------------------- #
# gauge_damping! is an exact gauge symmetry (this is what fixes the propagated
# derivatives). In particular the action must be invariant at *every* order:
# that only holds because the pseudofermion transforms too (phi -> Lambda phi).
# --------------------------------------------------------------------------- #

NDER = 2
NS   = 1 + NDER
L    = 8

am0 = Series{ComplexF64, NS}(ntuple(
    i -> i == 1 ? ComplexF64(0.1) : (i == 2 ? 1.0+0im : 0.0+0im), NS))

model = U1Nf2(Float64, typeof(am0), beta = 4.0, am0 = am0, iL = (L, L), BC = PeriodicBC)

# config with nonzero derivative orders, so gauge damping is nontrivial
Random.seed!(7)
m0 = U1Nf2(Float64, beta = 4.0, am0 = 0.1, iL = (L, L), BC = PeriodicBC)
LFTU1.randomize!(m0)
for i in eachindex(m0.U)
    model.U[i] = Series{ComplexF64, NS}(ntuple(k -> k == 1 ? m0.U[i] : (0.2im * m0.U[i]) / k, NS))
end

gd = U1Nf2SMDgd(model, SMD(integrator = OMF4(1.0, 4), gamma = 4.0); lambda0 = 1.5)
generate_pseudofermions!(model, gd.smd)

S_before  = LFTU1.action(model, gd.smd.hmcws)
U0_before = [model.U[i].c[1] for i in eachindex(model.U)]

gauge_damping!(model, gd, 0.25, exp(-1.0))

S_after  = LFTU1.action(model, gd.smd.hmcws)
U0_after = [model.U[i].c[1] for i in eachindex(model.U)]

dS_orders = abs.(S_after.c .- S_before.c)          # per-order action change
U0_inv    = maximum(abs.(U0_after .- U0_before))   # change of the physical field
U0_unit   = maximum(abs.(abs.(U0_after) .- 1))     # deviation from the U(1) circle

@testset "gauge_damping! is an exact gauge symmetry" begin
    @test maximum(dS_orders) < 1e-8    # action invariant at every order (phi -> L phi)
    @test U0_inv  < 1e-12              # U^(0) left exactly fixed (Lambda^(0) = 1)
    @test U0_unit < 1e-12              # config stays on the U(1) circle
end

# --------------------------------------------------------------------------- #
# Statistical: over a run the plain SMD lets ||A^(1)|| (the order-1 gauge
# potential = mass-derivative of the field) diverge, while gauge damping bounds
# it. The separation is orders of magnitude, so the thresholds are generous.
# --------------------------------------------------------------------------- #

am0_1 = Series{ComplexF64, 2}(ntuple(i -> i == 1 ? ComplexF64(0.1) : 1.0+0im, 2))
mkmodel() = U1Nf2(Float64, typeof(am0_1), beta = 4.0, am0 = am0_1, iL = (L, L), BC = PeriodicBC)

Random.seed!(11)
mstart = U1Nf2(Float64, beta = 4.0, am0 = 0.1, iL = (L, L), BC = PeriodicBC)
LFTU1.randomize!(mstart)

model_p = mkmodel()
model_g = mkmodel()
for i in eachindex(mstart.U)
    s = Series{ComplexF64, 2}((mstart.U[i], 0.0+0im))
    model_p.U[i] = s
    model_g.U[i] = s
end

smplr     = SMD(integrator = OMF4(1.0, 4), gamma = 4.0)
sampler_p = LFTSampling.sampler(model_p, smplr)                 # plain SMD
sampler_g = U1Nf2SMDgd(model_g, smplr; lambda0 = 1.0)          # gauge-damped SMD

a1norm(m) = sqrt(sum(i -> abs2(imag(m.U[i].c[2] * inv(m.U[i].c[1]))), eachindex(m.U)))

ntraj = 40
Random.seed!(101); for _ in 1:ntraj; sample!(model_p, sampler_p); end
Random.seed!(101); for _ in 1:ntraj; sample!(model_g, sampler_g); end

a1_plain  = a1norm(model_p)
a1_gauged = a1norm(model_g)

@testset "gauge damping bounds the derivative zero mode" begin
    @test a1_plain  > 50                # plain SMD: longitudinal mode grows large
    @test a1_gauged < a1_plain / 10     # gauge damping: dramatically smaller
end
