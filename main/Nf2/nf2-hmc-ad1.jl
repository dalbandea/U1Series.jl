# ===========================================================================
# U(1) Nf=2 HMC with a FormalSeries-valued mass: the trajectory carries the
# mass-derivatives (up to order NDER) of the gauge field through the HMC.
#
# Structured like LFTU1/main/Nf2/nf2-main.jl, driven by a TOML infile. The
# only extra Model param is NDER (number of mass derivatives to propagate).
#
# All Series machinery (scalar->Series, narrowing real, randn(Series),
# cross-type convert, and the triangular invert!) lives in U1Series
# (src/U1SeriesHMC.jl); I/O for Series configs lives in src/U1SeriesIO.jl.
# ===========================================================================

import Pkg
Pkg.activate(".")
using Revise
using TOML

length(ARGS) == 1 || error("Only one argument is expected! (Path to input file)")
isfile(ARGS[1]) || error("Path provided is not a file")
infile = ARGS[1]

pdata = TOML.parsefile(infile)

using LFTSampling
using LFTU1
using FormalSeries
using U1Series
using Dates
using Logging

devstr = pdata["Model params"]["device"]
if devstr == "CUDA"
    import CUDA
    device = CUDA.device()
elseif devstr == "CPU"
    device = LFTU1.KernelAbstractions.CPU()
else
    error("Only acceptable devices are CUDA or CPU")
end

function create_simulation_directory(wdir::String, savename::String)
    configfile = joinpath(wdir, savename * ".bdio")
    mkpath(wdir)
    dst = joinpath(wdir, basename(infile))
    if abspath(infile) != abspath(dst)          # skip if the infile already lives in wdir
        cp(infile, dst, force = true)
    end
    return configfile
end

import LFTSampling: metropolis_accept_reject!
function LFTSampling.metropolis_accept_reject!(lftws::L, lftcp::L,
                                               dS::FormalSeries.Series) where {L <: LFTU1.U1}
    @info "dH = $dS"
    return nothing
end

# ---------------------------------------------------------------------------
# Read parameters
# ---------------------------------------------------------------------------

# Model params
beta  = pdata["Model params"]["beta"]
mass  = pdata["Model params"]["mass"]
lsize = pdata["Model params"]["L"]
tsize = pdata["Model params"]["T"]
BC    = eval(Meta.parse(pdata["Model params"]["BC"]))
NDER  = pdata["Model params"]["NDER"]          # number of mass derivatives

# HMC params
tau        = pdata["HMC params"]["tau"]
nsteps     = pdata["HMC params"]["nsteps"]
ntherm     = pdata["HMC params"]["ntherm"]
ntraj      = pdata["HMC params"]["ntraj"]
discard    = pdata["HMC params"]["discard"]
integrator = eval(Meta.parse(pdata["HMC params"]["integrator"]))

# Working directory
wdir     = pdata["Working directory"]["wdir"]
savename = pdata["Working directory"]["savename"]
cntinue  = pdata["Working directory"]["continue"]
cntfile  = pdata["Working directory"]["cntfile"]

# Series-valued mass: value `mass`, first-derivative direction = 1
am0 = Series{ComplexF64, 1 + NDER}(ntuple(
    i -> i == 1 ? ComplexF64(mass) : (i == 2 ? 1.0 + 0im : 0.0 + 0im), 1 + NDER))

# ---------------------------------------------------------------------------
# Build / load the Series-valued model
# ---------------------------------------------------------------------------

if cntinue == true
    @info "Reading from old simulation"
    configfile = cntfile
    ncfgs = LFTSampling.count_configs(configfile)
    fb, model_s = read_cnfg_info(configfile, U1Nf2, Series{ComplexF64, 1 + NDER})
    LFTSampling.read_cnfg_n(fb, ncfgs, model_s)
    close(fb)
else
    @info "Creating simulation directory"
    ncfgs = 0

    model_s = U1Nf2(Float64, typeof(am0),
                    beta = beta,
                    am0  = am0,
                    iL   = (lsize, tsize),
                    BC   = BC,
                    device = device,
                   )

    # Random hotstart config in the constant order (zero derivative orders):
    # randomize a Float64 model and copy its links into model_s.
    model0 = U1Nf2(Float64,
                   beta = beta,
                   am0  = mass,
                   iL   = (lsize, tsize),
                   BC   = BC,
                   device = device,
                  )
    randomize!(model0)
    model_s.U .= model0.U

    configfile = create_simulation_directory(wdir, savename)
end

smplr     = HMC(integrator = integrator(tau, nsteps))
samplerws = LFTSampling.sampler(model_s, smplr)

logio = open(joinpath(wdir, savename * "_log.txt"), "a+")
logger = SimpleLogger(logio)
global_logger(logger)

@info "U(1) NF=2 SERIES SIMULATION (NDER = $NDER)" model_s.params smplr

# ---------------------------------------------------------------------------
# Thermalization
# ---------------------------------------------------------------------------

if cntinue == true
    @info "Skipping thermalization"
else
    @info "Starting thermalization"
    for i in 1:ntherm
        @info "THERM STEP $i"
        @time sample!(model_s, samplerws)
        flush(logio)
    end
end

# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

if cntinue == true
    @info "Restarting simulation from trajectory $ncfgs"
else
    @info "Starting simulation"
end

@time for i in (ncfgs + 1):(ncfgs + ntraj)
    @info "TRAJECTORY $i"
    for j in 1:discard
        @time sample!(model_s, samplerws)
    end
    @time sample!(model_s, samplerws)
    save_cnfg(configfile, model_s)
    flush(logio)
end

@info "Simulation finished succesfully"
flush(logio)
close(logio)
