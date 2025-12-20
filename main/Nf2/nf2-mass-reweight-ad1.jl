using Revise
import Pkg
Pkg.activate(".")
using LFTSampling
using LFTU1
using FormalSeries
using U1Series
using ProgressBars
using HDF5
using ArgParse
using LinearAlgebra


parse_commandline() = parse_commandline(ARGS)
function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "-L"
        help = "lattice size"
        required = true
        arg_type = Int

        "--start"
        help = "start from configuration"
        required = false
        arg_type = Int
        default = 1

        "--nconf"
        help = "number of configurations to analyze; 0 means until the end"
        required = false
        arg_type = Int
        default = 0

        "--nsrc"
        help = "number of sources"
        required = false
        arg_type = Int
        default = 2

        "--nder"
        help = "number of derivatives to compute"
        required = true
        arg_type = Int

        "--ens"
        help = "path to ensemble with configurations"
        required = true
        arg_type = String
        # default = "configs/"

        "--wdir"
        help = "path where to save data"
        required = true
        arg_type = String
    end
    return parse_args(args, s)
end


function save_data(Wvec, dirpath, cnfg)
    fname = joinpath(dirpath,"measurements/2pt-stoc-reweight_n$cnfg.h5")
    fid = h5open(fname, "w")
    write(fid, "reweighting_factors", series_stack(Wvec))
    close(fid)
    return nothing
end

# args = [
# "-L", "24",
# "--ens", "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/trash/Nf2sim-b4.0-L24-m0.02_D2024-04-12-16-28-46.31/Nf2sim-b4.0-L24-m0.02_D2024-04-12-16-28-46.31.bdio",
# "--start", "2",
# "--nconf", "10",
# "--nsrc", "2",
# "--nder", "3",
# ]
# parsed_args = parse_commandline(args)

parsed_args = parse_commandline(ARGS)

const NFL = 2  # number of flavors, hardcoded to be 2 by now
const N0 = parsed_args["L"]
const NSRC = parsed_args["nsrc"]
const NDER = parsed_args["nder"]

cfile = parsed_args["ens"]
isfile(cfile) || error("Path provided is not a file")
wdir = parsed_args["wdir"]
isdir(wdir) || error("wdir $wdir does not exist")

start = parsed_args["start"]
ncfgs = parsed_args["nconf"]
if ncfgs == 0
    ncfgs = LFTSampling.count_configs(cfile) - start + 1
end
finish = start + ncfgs - 1

fb, model = read_cnfg_info(cfile, U1Nf2; v = "0.1") # seems to be for an old format config


# Create am0 from params of model
am0 = Series{ComplexF64, 1+NDER}(ntuple(i -> begin
                                          if i == 1
                                              model.params.am0 + 0.0im
                                          elseif i == 2
                                              1.0 + 0.0im
                                          else
                                              0.0 + 0.0im
                                          end
                                      end, 1+NDER))

model_s = U1Nf2(Float64,
                typeof(am0),
                beta = model.params.beta,
                am0 = am0,
                iL = model.params.iL,
                BC = model.params.BC,
               )



lp = model.params
pws = U1Correlator(model_s)
Ws = Vector{typeof(am0)}(undef, NSRC)

for i in ProgressBar(start:finish)
    if i == start && start != 1
        LFTSampling.read_cnfg_n(fb, start, model)
    else
        read_next_cnfg(fb, model)
    end
    model_s.U .= model.U
    Ws .= 0.0
    for j in 1:NSRC
        pws.S0[:,:,:] .= randn(ComplexF64, lp.iL[1], lp.iL[2] ,2)
        LFTU1.invert!(pws.S, LFTU1.gamm5Dw_sqr_msq_am0!(model_s.params.am0), pws.S0, model_s.sws, model_s)
        LFTU1.gamm5Dw_sqr_msq_am0!(model.params.am0)(model_s.sws.Ap, model_s.sws.tmp, pws.S, model)
        Ws[j] = exp(-dot(pws.S0, model_s.sws.Ap) + dot(pws.S0, pws.S0))
    end
    save_data(Ws, wdir, i)
end
close(fb)
