####################################################################
# I/O for U1 configurations whose gauge field U (and mass) are
# FormalSeries-valued. Generalizes LFTU1/src/U1io.jl in the same way
# QuantumRotorExperiments/src/qrio.jl generalizes the quantum rotor:
# the Series link data is serialized order by order.
#
# The plain-Float64 U1 I/O in LFTU1 is left untouched; the methods here
# are strictly more specific (they dispatch on U being Series-valued),
# so `save_cnfg`/`read_next_cnfg` pick them up automatically.
####################################################################

import BDIO: BDIO_write!, BDIO_read
import LFTSampling: save_cnfg_header, read_cnfg_info
import FormalSeries: Series

# Number of stored orders of a Series type
_series_N(::Type{<:Series{T,N}}) where {T,N} = N

# ------------------------------------------------------------------ #
# Series array (config data) — one Series per lattice link
# ------------------------------------------------------------------ #

# Complex coefficients: store (real, imag) of every order
function BDIO.BDIO_write!(fb::BDIO.BDIOstream, U::Array{Series{T,N}}) where {T <: Complex, N}
    for i in eachindex(U), n in 1:N
        BDIO.BDIO_write!(fb, [real(U[i][n])])
        BDIO.BDIO_write!(fb, [imag(U[i][n])])
    end
    return nothing
end

function BDIO.BDIO_read(fb::BDIO.BDIOstream, U::Array{Series{T,N}}) where {T <: Complex, N}
    reg = zeros(Float64, 2*N)
    for i in eachindex(U)
        BDIO.BDIO_read(fb, reg)
        U[i] = Series{T,N}(ntuple(n -> complex(reg[2*n-1], reg[2*n]), N))
    end
    return nothing
end

# Real coefficients: store every order
function BDIO.BDIO_write!(fb::BDIO.BDIOstream, U::Array{Series{T,N}}) where {T <: Real, N}
    for i in eachindex(U), n in 1:N
        BDIO.BDIO_write!(fb, [U[i][n]])
    end
    return nothing
end

function BDIO.BDIO_read(fb::BDIO.BDIOstream, U::Array{Series{T,N}}) where {T <: Real, N}
    reg = zeros(Float64, N)
    for i in eachindex(U)
        BDIO.BDIO_read(fb, reg)
        U[i] = Series{T,N}(ntuple(n -> reg[n], N))
    end
    return nothing
end

# ------------------------------------------------------------------ #
# Model-level read/write for a Series-valued U1Nf2 workspace
# ------------------------------------------------------------------ #

function BDIO.BDIO_write!(fb::BDIO.BDIOstream,
                          u1ws::U1Nf2workspace{PRC, <:AbstractArray{<:Series}}) where {PRC}
    BDIO.BDIO_write!(fb, u1ws.U)
    return nothing
end

function BDIO.BDIO_read(fb::BDIO.BDIOstream,
                        u1ws::U1Nf2workspace{PRC, <:AbstractArray{<:Series}}) where {PRC}
    BDIO.BDIO_read(fb, u1ws.U)
    return nothing
end

# ------------------------------------------------------------------ #
# Header: like the Float64 U1Nf2 header, but the mass is a Series and
# is stored order by order, preceded by the number of orders N.
# ------------------------------------------------------------------ #

function save_cnfg_header(fb::BDIO.BDIOstream,
                          u1ws::U1Nf2workspace{PRC, <:AbstractArray{<:Series}}) where {PRC}
    BC = u1ws.params.BC == PeriodicBC ? 0 : 1
    N  = _series_N(eltype(u1ws.U))
    BDIO.BDIO_write!(fb, [u1ws.params.beta])
    BDIO.BDIO_write!(fb, [convert(Int32, N)])
    BDIO.BDIO_write!(fb, [u1ws.params.am0])         # Series mass -> 2N (or N) floats
    BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[1])])
    BDIO.BDIO_write!(fb, [convert(Int32, u1ws.params.iL[2])])
    BDIO.BDIO_write!(fb, [convert(Int32, BC)])
    BDIO.BDIO_write_hash!(fb)
    return nothing
end

"""
    read_cnfg_info(fname::String, ::Type{U1Nf2}, ::Type{Series{T,N}})

Reads theory parameters of a Series-valued U(1) Nf=2 ensemble from `fname` and
returns `(fb, model)` with `model` a `U1Nf2` whose links / mass are
`Series{T,N}`-valued. The link Series type must be supplied so the reader knows
how many orders to expect (it is cross-checked against the stored value).
"""
function read_cnfg_info(fname::String, ::Type{U1Nf2}, ::Type{Series{T,N}}) where {T, N}

    fb = BDIO.BDIO_open(fname, "r")

    while BDIO.BDIO_get_uinfo(fb) != 1
        BDIO.BDIO_seek!(fb)
    end

    ffoo = Vector{Float64}(undef, 1)
    BDIO.BDIO_read(fb, ffoo)
    beta = ffoo[1]

    ifoo = Vector{Int32}(undef, 1)
    BDIO.BDIO_read(fb, ifoo)
    Nstored = convert(Int64, ifoo[1])
    Nstored == N || error("Stored series order ($Nstored) differs from requested ($N)")

    amfoo = [zero(Series{T,N})]
    BDIO.BDIO_read(fb, amfoo)
    am0 = amfoo[1]

    ifoo = Vector{Int32}(undef, 3)
    BDIO.BDIO_read(fb, ifoo)
    lsize1 = convert(Int64, ifoo[1])
    lsize2 = convert(Int64, ifoo[2])
    BC     = convert(Int64, ifoo[3])
    BCt    = BC == 0 ? PeriodicBC : OpenBC

    model = U1Nf2(Float64, Series{T,N},
                  beta = beta,
                  am0  = am0,
                  iL   = (lsize1, lsize2),
                  BC   = BCt,
                 )

    return fb, model
end

"""
    read_cnfg_info(fname::String, ::Type{U1Nf2}, ::Type{Series})

Same as the method above, but the number of orders `N` is read from the file
header instead of being supplied by the caller, so one can write simply

    fb, model = read_cnfg_info(fname, U1Nf2, Series)

The link coefficient type is taken to be `ComplexF64` (U(1) links are always
complex). Note the returned model type is only known at run time (this call is
deliberately type-unstable), which is fine for configuration I/O.
"""
function read_cnfg_info(fname::String, ::Type{U1Nf2}, ::Type{Series})
    # peek the stored number of orders N from the header, then delegate
    fb = BDIO.BDIO_open(fname, "r")
    while BDIO.BDIO_get_uinfo(fb) != 1
        BDIO.BDIO_seek!(fb)
    end
    BDIO.BDIO_read(fb, Vector{Float64}(undef, 1))     # beta (skipped)
    ifoo = Vector{Int32}(undef, 1)
    BDIO.BDIO_read(fb, ifoo)
    N = convert(Int64, ifoo[1])
    BDIO.BDIO_close!(fb)

    return read_cnfg_info(fname, U1Nf2, Series{ComplexF64, N})
end

"""
    read_ensemble(fname::String, ::Type{U1Nf2}, ST::Type{<:Series}, n::Int64 = 0)

Reads a Series-valued U1Nf2 ensemble (link type `ST = Series{T,N}`) from `fname`.
If `n>0`, only the first `n` configurations are returned.
"""
function read_ensemble(fname::String, ::Type{U1Nf2}, ::Type{Series{T,N}}, n::Int64 = 0) where {T, N}
    nc = LFTSampling.count_configs(fname)
    fb, model = read_cnfg_info(fname, U1Nf2, Series{T,N})

    if n > 0
        n <= nc || error("Requested $n configurations but file has only $nc")
        nc = n
    end

    ens = [deepcopy(model) for _ in 1:nc]
    for i in 1:nc
        LFTSampling.read_next_cnfg(fb, ens[i])
        print("Reading configuration $i / $nc\r")
    end
    BDIO.BDIO_close!(fb)

    return ens
end
