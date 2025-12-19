
import LFTSampling: stopping_criterium

function LFTSampling.stopping_criterium(resnorm, res::Array{Series{T,N}}, tol::Series) where {T,N}
    sresnorm = compute_resnorm(res)
    return (real.(sresnorm.c) .< real((tol).c[1])) |> prod
end

function compute_resnorm(res::Array{Series{T,N}}) where {T,N}
    return Series{T,N}(ntuple(j -> sqrt(sum([abs2(item.c[j]) for item in res])), N))
end
