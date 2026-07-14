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

# --------------------------------------------------------------------------- #
# OpenBC. The link + pseudofermion transformation is BC-independent, so the exact
# gauge symmetry (action invariant at every order, U^(0) and unitarity fixed) must
# hold under OpenBC too -- this checks the OpenBC code path. The only BC-specific
# piece is the omega-update stencil: under OpenBC the wrap-around links are masked
# out of the divergence, which we test directly below.
# --------------------------------------------------------------------------- #

Lo = 8

am0_o = Series{ComplexF64, NS}(ntuple(
    i -> i == 1 ? ComplexF64(0.1) : (i == 2 ? 1.0+0im : 0.0+0im), NS))

model_o = U1Nf2(Float64, typeof(am0_o), beta = 4.0, am0 = am0_o, iL = (Lo, Lo), BC = OpenBC)

Random.seed!(23)
mo = U1Nf2(Float64, beta = 4.0, am0 = 0.1, iL = (Lo, Lo), BC = OpenBC)
LFTU1.randomize!(mo)
for i in eachindex(mo.U)
    model_o.U[i] = Series{ComplexF64, NS}(ntuple(k -> k == 1 ? mo.U[i] : (0.2im * mo.U[i]) / k, NS))
end

gd_o = U1Nf2SMDgd(model_o, SMD(integrator = OMF4(1.0, 4), gamma = 4.0); lambda0 = 1.5)
generate_pseudofermions!(model_o, gd_o.smd)

So_before  = LFTU1.action(model_o, gd_o.smd.hmcws)
U0o_before = [model_o.U[i].c[1] for i in eachindex(model_o.U)]

gauge_damping!(model_o, gd_o, 0.25, exp(-1.0))

So_after  = LFTU1.action(model_o, gd_o.smd.hmcws)
U0o_after = [model_o.U[i].c[1] for i in eachindex(model_o.U)]

dSo_orders = abs.(So_after.c .- So_before.c)
U0o_inv    = maximum(abs.(U0o_after .- U0o_before))
U0o_unit   = maximum(abs.(abs.(U0o_after) .- 1))

@testset "OpenBC: gauge_damping! is an exact gauge symmetry" begin
    @test maximum(dSo_orders) < 1e-8    # action invariant at every order (phi -> L phi)
    @test U0o_inv  < 1e-12              # U^(0) left exactly fixed (Lambda^(0) = 1)
    @test U0o_unit < 1e-12              # config stays on the U(1) circle
end

# The OpenBC omega update must ignore the wrap-around links (x-links at i1=Nx,
# y-links at i2=Ny): two configs that differ ONLY on those links -- in their
# derivative orders, U^(0) untouched -- must yield the *same* omega after one
# damping step. Without the mask their (here deliberately large) derivatives would
# leak into d*A.
mkmodel_o() = U1Nf2(Float64, typeof(am0_o), beta = 4.0, am0 = am0_o, iL = (Lo, Lo), BC = OpenBC)
model_a = mkmodel_o()
model_b = mkmodel_o()

Random.seed!(29)
mbase = U1Nf2(Float64, beta = 4.0, am0 = 0.1, iL = (Lo, Lo), BC = OpenBC)
LFTU1.randomize!(mbase)
for i in eachindex(mbase.U)
    s = Series{ComplexF64, NS}(ntuple(k -> k == 1 ? mbase.U[i] : (0.2im * mbase.U[i]) / k, NS))
    model_a.U[i] = s
    model_b.U[i] = s
end

# perturb ONLY the wrap-around links' derivative orders in model_b (U^(0) kept)
bump(u) = Series{ComplexF64, NS}(ntuple(k -> k == 1 ? u.c[1] : (5.0im * u.c[1]) / k, NS))
for i2 in 1:Lo; model_b.U[Lo, i2, 1] = bump(model_b.U[Lo, i2, 1]); end   # x-links at i1=Nx
for i1 in 1:Lo; model_b.U[i1, Lo, 2] = bump(model_b.U[i1, Lo, 2]); end   # y-links at i2=Ny

gd_a = U1Nf2SMDgd(model_a, SMD(integrator = OMF4(1.0, 4), gamma = 4.0); lambda0 = 1.5)
gd_b = U1Nf2SMDgd(model_b, SMD(integrator = OMF4(1.0, 4), gamma = 4.0); lambda0 = 1.5)

gauge_damping!(model_a, gd_a, 0.25, exp(-1.0))
gauge_damping!(model_b, gd_b, 0.25, exp(-1.0))

omega_diff = maximum(maximum(abs.(gd_a.omega[i].c .- gd_b.omega[i].c)) for i in eachindex(gd_a.omega))

@testset "OpenBC: omega update ignores the wrap-around links" begin
    @test omega_diff < 1e-12
end
