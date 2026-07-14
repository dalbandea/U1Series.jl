####################################################################
# Gauge damping for the Series-valued U(1) Nf=2 SMD (NSPT).
#
# Specialisation to U(1) of Dalla Brida & Lüscher, arXiv:1703.04396 (sec. 5.4):
# the mass term is gauge invariant, so the propagated mass-derivatives U^(k),
# k>=1, have a longitudinal (pure-gauge) component that the force annihilates and
# that therefore random-walks in Monte-Carlo time. A MC-time-dependent gauge
# transformation damps it away.
#
# This is deliberately kept SEPARATE from the plain SMD (src/U1SeriesSMD.jl): the
# gauge-damped sampler U1Nf2SMDgd just *wraps* a tested U1Nf2SMD and reuses its
# refresh_momenta! / molecular_dynamics! / generate_pseudofermions! untouched; the
# only new ingredient is gauge_damping!. Nothing in the working SMD/HMC path is
# modified.
#
# Why U(1) is easy (abelian):
#   * the momenta are inert under the gauge transformation, so only the links move
#   * on the phase Theta_mu = -i log U_mu the transformation is additive and
#     decoupled order by order:  Theta_mu(x) -> Theta_mu(x) - eps (d omega)_mu(x)
#   * keeping omega with zero constant order leaves U^(0) *exactly* unchanged.
#
# Per SMD cycle (step eps, same c1 = exp(-gamma eps) as the momentum refresh):
#   A       = -i log(U / U^(0))                 derivative gauge potential, A^(0)=0
#   omega   = c1 omega + eps*lambda0 (d* A)     (d* = lattice divergence)
#   Lambda  = exp(i eps omega)                  (Lambda^(0) = 1)
#   U_mu(x) = Lambda(x) U_mu(x) conj(Lambda(x+mu))
#
# Implemented for PeriodicBC (the d / d* stencils use circshift).
####################################################################

import LFTSampling: sample!, refresh_momenta!, molecular_dynamics!,
                    generate_pseudofermions!, Hamiltonian, metropolis_accept_reject!
import FormalSeries: Series

export U1Nf2SMDgd, gauge_damping!

# Gauge-damped SMD workspace: a plain SMD workspace + the auxiliary field omega.
struct U1Nf2SMDgd{S <: U1Nf2SMD, A <: AbstractArray}
    smd::S              # tested plain-SMD workspace (momenta, pseudofermions, forces)
    omega::A            # gauge-damping field (real Series, one per site, omega^(0)=0)
    lambda0::Float64    # gauge-damping strength
end

function U1Nf2SMDgd(u1ws::LFTU1.U1Nf2, smdp::SMDParams; lambda0::Real = 1.0)
    smd   = U1Nf2SMD(u1ws, smdp)
    omega = LFTU1.to_device(u1ws.device, zeros(real(eltype(u1ws.U)), u1ws.params.iL...))
    return U1Nf2SMDgd(smd, omega, Float64(lambda0))
end

# ---- small truncated-Series helpers (arguments have zero constant order) ------

# log(1+n) = n - n^2/2 + n^3/3 - ...  (finite: n^j has lowest order j)
function series_log1p(n::Series{T, N}) where {T, N}
    acc = zero(n)
    p   = n
    for j in 1:N-1
        acc = acc + ((-1.0)^(j + 1) / j) * p
        p   = p * n
    end
    return acc
end

# exp(z) = 1 + z + z^2/2 + ...  (finite: z^j has lowest order j)
function series_exp0(z::Series{T, N}) where {T, N}
    acc = Series{T, N}(1)
    p   = z
    f   = 1.0
    for j in 1:N-1
        f  *= j
        acc = acc + (1.0 / f) * p
        p   = p * z
    end
    return acc
end

# derivative gauge potential of one link:  A = -i log(U / U^(0))  (real Series)
function deriv_potential(u::Series{CT, N}) where {CT, N}
    inv0 = inv(u.c[1])                                   # 1 / U^(0)
    R    = Series{CT, N}(ntuple(k -> u.c[k] * inv0, N))  # R^(0) = 1
    L    = series_log1p(R - Series{CT, N}(1))            # coeffs are i*theta^(k)
    FT   = real(CT)
    return Series{FT, N}(ntuple(k -> imag(L.c[k]), N))   # theta^(k), real, theta^(0)=0
end

# gauge transformation of one site:  Lambda = exp(i eps omega)   (Lambda^(0) = 1)
function gauge_link(w::Series{FT, N}, eps) where {FT, N}
    CT = complex(FT)
    z  = Series{CT, N}(ntuple(k -> CT(im * eps * w.c[k]), N))
    return series_exp0(z)
end

conj_series(g::Series{T, N}) where {T, N} = Series{T, N}(conj.(g.c))

# ---- one gauge-damping step --------------------------------------------------

function gauge_damping!(u1ws::LFTU1.U1Nf2, gd::U1Nf2SMDgd, eps, c1)
    u1ws.params.BC == PeriodicBC ||
        error("gauge_damping! is implemented for PeriodicBC only")
    U     = u1ws.U
    omega = gd.omega

    # derivative gauge potential A_mu(x) (real Series, A^(0) = 0)
    A  = deriv_potential.(U)
    A1 = @view A[:, :, 1]
    A2 = @view A[:, :, 2]

    # lattice divergence  (d* A)(x) = sum_mu [ A_mu(x-mu) - A_mu(x) ]
    dstarA = (circshift(A1, (1, 0)) .- A1) .+ (circshift(A2, (0, 1)) .- A2)

    # omega <- c1 omega + eps*lambda0 (d* A)   (stays zero constant order)
    omega .= c1 .* omega .+ (eps * gd.lambda0) .* dstarA

    # gauge transformation, U^(0) left unchanged since Lambda^(0) = 1:
    #   links:          U_mu(x)   -> Lambda(x) U_mu(x) conj(Lambda(x+mu))
    #   pseudofermion:  phi(x)    -> Lambda(x) phi(x)
    # The pseudofermion must transform too: S_pf[U,phi] is gauge invariant only if
    # phi rotates with U (D[U^L] = L D[U] L^dag). Without it the derivative orders
    # of S_pf are not invariant and the trajectory injects spurious energy.
    Lam  = gauge_link.(omega, eps)
    cLam = conj_series.(Lam)                       # Lambda^{-1} = conj (unitary)
    U[:, :, 1] .= Lam .* @view(U[:, :, 1]) .* circshift(cLam, (-1, 0))
    U[:, :, 2] .= Lam .* @view(U[:, :, 2]) .* circshift(cLam, (0, -1))

    F = gd.smd.hmcws.F                             # pseudofermion field phi
    F[:, :, 1] .= Lam .* @view(F[:, :, 1])
    F[:, :, 2] .= Lam .* @view(F[:, :, 2])
    return nothing
end

# ---- one gauge-damped SMD trajectory (standard SMD = GHMC) --------------------
# Same GHMC skeleton as the plain SMD sample! (partial refresh once -> MD
# trajectory -> dH -> accept/reject), with gauge_damping! appended to each MD
# step. All sampling primitives are the tested plain-SMD ones (called on the
# wrapped workspace); only gauge_damping! is new. dH is @info'd via the
# (skipped-for-Series) accept/reject, as in the plain SMD / HMC.
function LFTSampling.sample!(u1ws::LFTU1.U1Nf2, gd::U1Nf2SMDgd)
    smd    = gd.smd
    ws_cp  = deepcopy(u1ws)
    generate_pseudofermions!(u1ws, smd)

    integr = smd.params.integrator
    eps    = integr.epsilon
    c1     = exp(-smd.params.gamma * eps)

    refresh_momenta!(u1ws, smd, c1)                   # single partial refresh
    Hini = Hamiltonian(u1ws, smd.hmcws)
    for _ in 1:integr.nsteps                          # MD trajectory, no mid refresh
        molecular_dynamics!(u1ws, smd)
        gauge_damping!(u1ws, gd, eps, c1)
    end
    dH = Hamiltonian(u1ws, smd.hmcws) - Hini

    metropolis_accept_reject!(u1ws, ws_cp, dH)
    return nothing
end
