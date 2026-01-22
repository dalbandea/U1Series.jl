
import ADerrors: uwreal
import Base: zero

zero(::Type{uwreal}) = uwreal([0.0,0.0], "zero")

"""
Transforms a vector of FormalSeries.Series into an uwreal FormalSeries.Series
"""
function ADerrors.uwreal(obs::Vector{FormalSeries.Series{T, N}}, args...) where {T, N}
    uwobs = FormalSeries.Series{ADerrors.uwreal, N}(ntuple(i -> ADerrors.uwreal([real(obs[j].c[i]) for j in 1:length(obs)], args...), N))
    return uwobs
end

function ADerrors.uwerr(obs::FormalSeries.Series{ADerrors.uwreal, N}) where N
    for i in 1:N
        ADerrors.uwerr(obs[i])
    end
end

