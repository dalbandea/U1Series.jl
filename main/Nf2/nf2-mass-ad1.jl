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

# args = [
# "-L", "24",
# "--ens", "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/trash/Nf2sim-b4.0-L24-m0.02_D2024-04-12-16-28-46.31/Nf2sim-b4.0-L24-m0.02_D2024-04-12-16-28-46.31.bdio",
# "--start", "2",
# "--nconf", "10",
# "--nsrc", "2",
# "--nder", "1",
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

#######################
# Compute correlators #
#######################

"""
- Compute connected traces and save them into data.P[ifl, jfl, t], already
  averaging over the number of sources. 
- Compute disconnected traces separately and save them to data.Delta[ifl, isrc, t]
""" 
function correlators(data, corrws, u1ws, nsrc)
    reset_data(data)
    for isrc in ProgressBar(1:nsrc)
        for it in 1:N0
            random_source(it,corrws,u1ws)
            for ifl in 1:2
                disconnected_correlator(corrws, u1ws, ifl)
                data.Delta[ifl,isrc,:] .+= corrws.result
                for jfl in ifl:2
                    connected_correlator(corrws, u1ws, ifl, jfl)
                    for t in 1:N0
                        tt=((t-it+N0)%N0+1);
                        data.P[ifl, jfl, tt] += corrws.result[t] ./ N0 ./ nsrc
                    end
                end
            end
        end
    end
end

function compute_disconnected!(data, nsrc)
    data.disc .= 0.0
    for ifl in 1:2, jfl in ifl:2
        for isrc in 1:nsrc, jsrc in 1:nsrc
            if jsrc != isrc
                for t in 1:N0, tt in 1:N0
                    data.disc[ifl, jfl, t] += data.Delta[ifl, isrc, tt] * data.Delta[jfl, jsrc, (tt+t-1-1)%N0+1] / N0 / nsrc / (nsrc - 1)
                end
            end
        end
    end
    return nothing
end

function reset_data(data)
    data.P .= 0.0
    data.Delta .= 0.0
    data.disc .= 0.0
    return nothing
end


function save_data(data, dirpath, cnfg)
    fname = joinpath(dirpath,"measurements/2pt-stoc-conn-disc_n$cnfg.h5")
    fid = h5open(fname, "w")
    write(fid, "connected", series_stack(data.P))
    write(fid, "disconnected", series_stack(data.disc))
    write(fid, "Delta", series_stack(data.Delta))
    close(fid)
    return nothing
end

data = (
    nc = 0,
    P = zeros(Series{ComplexF64, 1+NDER}, NFL, NFL, N0),
    disc = zeros(Series{ComplexF64, 1+NDER}, NFL, NFL, N0),
    Delta = zeros(Series{ComplexF64, 1+NDER}, NFL, NSRC, N0)
)

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

pws = U1Correlator(model_s, wdir=wdir)

for i in ProgressBar(start:finish)
    if i == start && start != 1
        LFTSampling.read_cnfg_n(fb, start, model_s)
    else
        read_next_cnfg(fb, model_s)
    end
    correlators(data, pws, model_s, NSRC)
    compute_disconnected!(data, NSRC)
    save_data(data, wdir, i)
end
close(fb)
