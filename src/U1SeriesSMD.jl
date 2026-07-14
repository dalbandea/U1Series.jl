####################################################################
# SMD (Stochastic Molecular Dynamics) support for the U(1) Nf=2 model,
# added the same minimal way LFTQuantumRotor does it (quantumrotorsmd.jl):
# a dedicated SMD sampler workspace plus the partial momentum refresh.
#
# Here the workspace is a thin wrapper around a U1Nf2HMC workspace, so the
# full HMC force machinery (force!, update_momenta!, generate_pseudofermions!,
# update_fields!) is reused untouched by delegating the MD-step primitives to
# the inner HMC workspace. This works for a plain Float64 model and, unchanged,
# for a FormalSeries-valued one (which is the point: SMD with mass derivatives).
#
# It lives in U1Series rather than in LFTU1 because LFTU1 is used un-dev'd here;
# these are all method *additions* (new signatures / a new type), so there is no
# piracy.
####################################################################

import LFTSampling: sampler, refresh_momenta!, update_momenta!, update_fields!,
                    generate_momenta!, generate_pseudofermions!, sample!,
                    molecular_dynamics!, Hamiltonian, metropolis_accept_reject!

# SMD workspace: wraps a U1Nf2HMC so the force machinery is reused as-is.
struct U1Nf2SMD{H <: LFTU1.U1Nf2HMC} <: AbstractSMD
    params::SMDParams
    hmcws::H
end

function U1Nf2SMD(u1ws::LFTU1.U1Nf2, smdp::SMDParams)
    hmcws = LFTU1.U1Nf2HMC(u1ws, HMC())     # inner HMC ws; its HMC params are unused
    generate_momenta!(u1ws, hmcws)          # SMD keeps momenta across trajectories
    return U1Nf2SMD{typeof(hmcws)}(smdp, hmcws)
end

LFTSampling.sampler(lftws::LFTU1.U1Nf2, smdp::SMDParams) = U1Nf2SMD(lftws, smdp)

# --- MD-step primitives: delegate to the inner (AbstractHMC) workspace --------

LFTSampling.update_momenta!(u1ws::LFTU1.U1Nf2, epsilon, smdws::U1Nf2SMD) =
    update_momenta!(u1ws, epsilon, smdws.hmcws)

LFTSampling.update_fields!(u1ws::LFTU1.U1Nf2, epsilon, smdws::U1Nf2SMD) =
    update_fields!(u1ws, epsilon, smdws.hmcws)

LFTSampling.generate_pseudofermions!(u1ws::LFTU1.U1Nf2, smdws::U1Nf2SMD) =
    generate_pseudofermions!(u1ws, smdws.hmcws)

# --- Partial momentum refresh -------------------------------------------------
# p <- c1*p + sqrt(1-c1^2)*eta, eta a Gaussian momentum with the same
# open/periodic BC handling as the HMC momenta heatbath.
function LFTSampling.refresh_momenta!(u1ws::LFTU1.U1Nf2, smdws::U1Nf2SMD, c1::Float64)
    lp  = u1ws.params
    mom = smdws.hmcws.mom
    eta = similar(mom)
    LFTU1.U1generate_momenta!(u1ws.device)(eta, lp.iL[1], lp.iL[2], lp.BC,
                                           ndrange = (lp.iL[1], lp.iL[2]),
                                           workgroupsize = u1ws.kprm.threads)
    LFTU1.KernelAbstractions.synchronize(u1ws.device)
    mom .= c1 .* mom .+ sqrt(1 - c1^2) .* eta
    return nothing
end

# --- One SMD trajectory (standard SMD = GHMC) ---------------------------------
# Same skeleton as HMC (hmc!), the only difference being a *partial* momentum
# refresh (c1 = exp(-gamma*eps)) instead of a full one:
#   heatbath pseudofermions -> partial refresh (once) -> MD trajectory ->
#   dH = H_final - H_initial -> accept/reject.
# The energy dH is exactly the quantity that enters the accept/reject, measured
# at the two ends of the trajectory as in HMC. For a FormalSeries-valued model
# the accept/reject is not differentiable and is skipped: the overridden
# metropolis_accept_reject!(::U1, ::U1, ::Series) just @info's dH (as in the HMC
# main file). On reject in the plain case a momentum flip should be added for
# exact GHMC detailed balance; that is not the FormalSeries use case here.
function LFTSampling.sample!(u1ws::LFTU1.U1Nf2, smdws::U1Nf2SMD)
    ws_cp  = deepcopy(u1ws)
    generate_pseudofermions!(u1ws, smdws)

    integr = smdws.params.integrator
    c1     = exp(-smdws.params.gamma * integr.epsilon)

    refresh_momenta!(u1ws, smdws, c1)                 # single partial refresh
    Hini = Hamiltonian(u1ws, smdws.hmcws)
    for _ in 1:integr.nsteps                          # MD trajectory, no mid refresh
        molecular_dynamics!(u1ws, smdws)
    end
    dH = Hamiltonian(u1ws, smdws.hmcws) - Hini

    metropolis_accept_reject!(u1ws, ws_cp, dH)
    return nothing
end
