####################################################################
# Series-through-HMC support: the FormalSeries helper methods and the
# triangular-inversion invert! needed to run the U1Nf2 HMC with a
# FormalSeries-valued mass / gauge field.
#
# invert!(X, gamm5Dw_sqr_msq!, F, model_s.sws, model_s) with Series-valued
# X, F and a Series-valued model dispatches here and solves A_s X = F order
# by order in the formal parameter, each order via an ordinary ComplexF64 CG
# on a constant-order companion model. See main/Nf2/nf2-hmc-ad1.jl.
####################################################################

import FormalSeries: Series
import Random: randn
import Base: real, convert
import Base.*

# ---- FormalSeries helpers -------------------------------------------------


# # This is in my devved version of FormalSeries already
Base.:*(s1::Series{T,N}, s2::Number) where {T,N} =
genseries(Series{typeof(promote(s1[1], s2)[1]),N}, i -> s1.c[i]*s2)

# scalar -> Series (value in the constant order, higher orders zero)
FormalSeries.Series{T, N}(x::Tx) where {T, N, Tx <: Number} =
    Series(ntuple(i -> i == 1 ? T(x) : zero(T), N))

# narrowing real:  Series{ComplexF64,N} -> Series{Float64,N}
# (FormalSeries' own `real` keeps the coefficient type)
Base.real(x::Series{ComplexF64, N}) where {N} =
    Series{Float64, N}(ntuple(i -> real(x.c[i]), N))

# one Gaussian draw, placed in the constant order
Random.randn(::Type{Series{T, N}}) where {T, N} = randn(T)

# convert between Series coefficient types, e.g. Float64 -> ComplexF64
Base.convert(::Type{Series{T, N}}, x::Series{S, N}) where {T, S, N} =
    Series{T, N}(ntuple(i -> convert(T, x.c[i]), N))
Series{T, N}(x::Series{S, N}) where {T, S, N} =
    Series{T, N}(ntuple(i -> convert(T, x.c[i]), N))

# ---- Triangular Series inversion ------------------------------------------

_norder(::Type{<:Series{T, N}}) where {T, N} = N
_coeff(::Type{<:Series{T, N}}) where {T, N} = T

# Constant-order (A_0) companion model, built lazily and refreshed each call.
const _CONST_COMPANION = Ref{Any}(nothing)

function _constant_companion(u1ws::U1Nf2)
    m0 = _CONST_COMPANION[]
    if m0 === nothing || m0.params.iL != u1ws.params.iL
        m0 = U1Nf2(Float64, ComplexF64,
                   beta = u1ws.params.beta,
                   am0  = real(u1ws.params.am0[1]),
                   iL   = u1ws.params.iL,
                   BC   = u1ws.params.BC)
        _CONST_COMPANION[] = m0
    end
    m0.U .= getindex.(u1ws.U, 1)               # constant-order gauge field
    return m0
end

"""
    invert!(so, A, si, solver::CG, u1ws::U1Nf2)

For a Series-valued output `so`, solves `A_s so = si` order by order in the
formal parameter. `A_s` is the operator `A(·, u1ws)` built from the Series-valued
model `u1ws` (its expansion coefficients `A_k` are extracted automatically by
FormalSeries arithmetic); the constant order `A_0` is solved with an ordinary CG
on a companion model built from the constant term of `u1ws.U`.
"""
function LFTSampling.invert!(so::AbstractArray{<:Series}, A::Function, si,
                             solver::CG, u1ws::U1Nf2)
    N    = _norder(eltype(so))
    CT   = _coeff(eltype(so))
    dims = size(so)

    m0   = _constant_companion(u1ws)
    am0c = real(u1ws.params.am0[1])

    Ax   = zeros(eltype(so), dims)             # A_s applied to a const-order field
    tmp  = zeros(eltype(so), dims)             # scratch for the squared operator
    rhs  = zeros(CT, dims)                      # rhs of the order-k system
    chi  = [zeros(CT, dims) for _ in 1:N]      # order-by-order solutions

    for k in 1:N
        # rhs_k = si_k - Σ_{j<k} A_{k-j} χ_j
        for i in eachindex(rhs)
            rhs[i] = si[i][k]
        end
        for j in 1:k-1
            A(Ax, tmp, chi[j], u1ws)
            for i in eachindex(rhs)
                rhs[i] -= Ax[i][k-j+1]
            end
        end
        LFTSampling.invert!(chi[k], LFTU1.gamm5Dw_sqr_msq_am0!(am0c), rhs, m0.sws, m0)
    end

    for i in eachindex(so)
        so[i] = Series{CT, N}(ntuple(k -> chi[k][i], N))
    end
    return nothing
end
