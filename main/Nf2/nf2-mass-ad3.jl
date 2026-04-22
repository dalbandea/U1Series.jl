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

import Base: float
Base.float(x::Series{T,N}) where {T,N} = Series(ntuple(i -> float(x[i]), N))

import FormalSeries: Series
FormalSeries.Series{T, N}(x::Tx) where {T,N,Tx<:Number} = Series(ntuple(i -> i == 1 ? T(x) : zero(T), N))

import LinearAlgebra: dot
# scalar interaction
dot(x::Series, y::Number) = conj(x) * y
dot(x::Number, y::Series) = conj(x) * y  # optional, depending on convention

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

        "--wdir"
        help = "path where to save data"
        required = true
        arg_type = String
    end
    return parse_args(args, s)
end


"""
Compute NDER+1 Float64 propagators via sequential Float64 CG solves, then assemble
Series-valued Rm[ifl]. The k-th solve gives the k-th mass derivative of (γ₅D)⁻¹η:
  sol[1]   = (γ₅D)⁻¹ η
  sol[k+1] = -[(γ₅D)⁻¹γ₅] sol[k]   (each application of (γ₅D)⁻¹ costs one CG inversion)
"""
function random_source_series!(Rm, t0, corrws, u1ws::U1Nf2, am0)
    S0 = corrws.S0
    S  = corrws.S
    lp = u1ws.params

    S0 .= zero(ComplexF64)
    S0[:,t0,:] .= randn(ComplexF64, lp.iL[1], 2)

    sols = [zeros(ComplexF64, size(S0)) for _ in 1:1+NDER]

    # 0th order: sol[1] = (γ₅D)⁻¹ S0 = D⁻¹γ₅S0
    invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(am0), S0, u1ws.sws, u1ws)
    gamm5Dw!(sols[1], S, am0, u1ws)

    # k-th order: sol[k+1] = -[(γ₅D)⁻¹γ₅] sol[k] = - D⁻¹sol[k]
    for k in 1:NDER
        LFTU1.gamm5!(S0, sols[k], u1ws)
        invert!(S, LFTU1.gamm5Dw_sqr_msq_am0!(am0), S0, u1ws.sws, u1ws)
        gamm5Dw!(sols[k+1], S, am0, u1ws)
        sols[k+1] .*= -1
    end

    # Both flavors have the same mass in U1Nf2
    for ifl in 1:NFL
        for j in eachindex(sols[1])
            Rm[ifl][j] = Series{ComplexF64, 1+NDER}(ntuple(k -> sols[k][j], 1+NDER))
        end
    end

    return nothing
end

function disconnected_correlator_series(Rm, S0, t, u1ws, ifl)
    lp = u1ws.params
    Ct = zero(eltype(Rm[ifl]))
    for x in 1:lp.iL[1]
        Ct += dot(S0[x,t,:], Rm[ifl][x,t,:]) / sqrt(lp.iL[1])
    end
    return Ct
end

function connected_correlator_series(Rm, t, u1ws, ifl, jfl)
    lp = u1ws.params
    Ct = zero(eltype(Rm[ifl]))
    for x in 1:lp.iL[1]
        Ct += dot(Rm[jfl][x,t,:], Rm[ifl][x,t,:]) / lp.iL[1]
    end
    return Ct
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

function correlators(data, Rm, corrws, u1ws, nsrc)
    reset_data(data)
    for isrc in ProgressBar(1:nsrc)
        for it in 1:N0
            random_source_series!(Rm, it, corrws, u1ws, am0)
            for ifl in 1:2
                for t in 1:N0
                    data.Delta[ifl,isrc,t] += disconnected_correlator_series(Rm, corrws.S0, t, u1ws, ifl)
                end
                for jfl in ifl:2
                    for t in 1:N0
                        tt = ((t-it+N0)%N0+1)
                        data.P[ifl, jfl, tt] += connected_correlator_series(Rm, t, u1ws, ifl, jfl) / N0 / nsrc
                    end
                end
            end
        end
    end
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


# Main body

# args = [
# "-L", "24",
# "--ens", "/home/david/git/dalbandea/phd/codes/6-LFTs/LFTModels/LFTU1.jl/trash/Nf2sim-b4.0-L24-m0.02_D2024-04-12-16-28-46.31/Nf2sim-b4.0-L24-m0.02_D2024-04-12-16-28-46.31.bdio",
# "--start", "2",
# "--nconf", "10",
# "--nsrc", "2",
# "--nder", "3",
# "--wdir", "trash/"
# ]
# parsed_args = parse_commandline(args)

parsed_args = parse_commandline(ARGS)

const NFL = 2
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

fb, model = read_cnfg_info(cfile, U1Nf2; v = "0.1")

# Use Float64 mass directly: no Series arithmetic inside the CG solver
const am0 = real(model.params.am0)

data = (
    nc = 0,
    P = zeros(Series{ComplexF64, 1+NDER}, NFL, NFL, N0),
    disc = zeros(Series{ComplexF64, 1+NDER}, NFL, NFL, N0),
    Delta = zeros(Series{ComplexF64, 1+NDER}, NFL, NSRC, N0)
)

pws = U1Correlator(model, wdir=wdir)

# Series-valued propagator fields, same shape as pws.R[ifl] but with Series elements
Rm = [zeros(Series{ComplexF64, 1+NDER}, size(pws.R[1])) for _ in 1:NFL]


for i in ProgressBar(start:finish)
    if i == start && start != 1
        LFTSampling.read_cnfg_n(fb, start, model)
    else
        read_next_cnfg(fb, model)
    end
    correlators(data, Rm, pws, model, NSRC)
    compute_disconnected!(data, NSRC)
    save_data(data, wdir, i)
end
close(fb)


# # Testing derivatives with nf2-mass-ad1.jl {{{
# LFTSampling.read_cnfg_n(fb, 1, model)

# import Random
# Random.seed!(1234)
# random_source_series!(Rm, 1, pws, model, am0)

# Rm[1][1]
# Rm[2][end]

# Random.seed!(1234)
# @time correlators(data, Rm, pws, model, NSRC)
# # }}}

