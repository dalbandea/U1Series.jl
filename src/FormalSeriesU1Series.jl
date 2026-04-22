import Base: complex

# Base.complex(::Type{Series{T,N}}) where {T,N} = Series{complex(T),N}
# Base.complex(s1::FormalSeries.Series, s2::FormalSeries.Series) = s1 + im*s2 # this only works if Series already complex...
# Base.complex(s1::FormalSeries.Series{T,N}) where {T,N} = s1 + im*zero(T)


Base.complex(::Type{Series{T,N}}) where {T,N} = Series{complex(T),N}
function Base.complex(s1::FormalSeries.Series, s2::FormalSeries.Series)
    (isreal(s1) && isreal(s2)) || error("Inputs $s1 or $s2 are already complex")
    return s1 + im*s2
end
Base.complex(s1::FormalSeries.Series{T,N}) where {T,N} = Series{complex(T),N}(ntuple(i -> complex(s1.c[i]), N))

import LinearAlgebra: inv, dot, transpose, adjoint
import Base: abs2, abs, isreal
Base.abs2(s1::FormalSeries.Series) = adjoint(s1)*s1
Base.abs(s1::FormalSeries.Series) = sqrt(abs2(s1))
Base.isreal(s::FormalSeries.Series) = prod(isreal.(s.c))
LinearAlgebra.inv(s::FormalSeries.Series{T,N}) where {T <: Real,N} = 1 / s
LinearAlgebra.dot(s1::FormalSeries.Series{T,N}, s2::FormalSeries.Series{T,N}) where {T,N} = conj(s1) * s2
LinearAlgebra.transpose(s::FormalSeries.Series{T,N}) where {T,N} = s
LinearAlgebra.adjoint(s::FormalSeries.Series{T,N}) where {T,N} = conj(s)



