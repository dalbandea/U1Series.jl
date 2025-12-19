module U1Series

using LFTSampling
using LFTU1
using FormalSeries
using EllipsisNotation
using HDF5

import Base: abs2, abs
Base.abs2(s1::Series) = adjoint(s1)*s1
Base.abs(s1::Series) = sqrt(abs2(s1))

include("LFTSamplingSeries.jl")

include("SeriesIO.jl")
export get_sorder, series_stack, series_unstack, save_data

end # module U1Series
