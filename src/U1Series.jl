module U1Series

import BDIO
import Random
using LFTSampling
using LFTU1
using FormalSeries
using EllipsisNotation
using HDF5
using LinearAlgebra
using ADerrors
using LeastSquaresOptim
using Statistics

include("LFTSamplingSeries.jl")

include("SeriesIO.jl")
export get_sorder, series_stack, series_unstack, save_data

include("U1SeriesIO.jl")

include("FormalSeriesU1Series.jl")

include("U1SeriesHMC.jl")

include("ADerrorsSeries.jl")

include("Fits.jl")
export fit_routine

end # module U1Series
