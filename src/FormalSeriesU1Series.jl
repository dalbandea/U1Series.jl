import Base: complex

Base.complex(s1::FormalSeries.Series, s2::FormalSeries.Series) = s1 + im*s2
Base.complex(s1::FormalSeries.Series{T,N}) where {T,N} = s1 + im*zero(T)

import LinearAlgebra: inv, dot, transpose, adjoint
import Base: abs2, abs
Base.abs2(s1::FormalSeries.Series) = adjoint(s1)*s1
Base.abs(s1::FormalSeries.Series) = sqrt(abs2(s1))
LinearAlgebra.inv(s::FormalSeries.Series{T,N}) where {T <: Real,N} = 1 / s
LinearAlgebra.dot(s1::FormalSeries.Series{T,N}, s2::FormalSeries.Series{T,N}) where {T,N} = conj(s1) * s2
LinearAlgebra.transpose(s::FormalSeries.Series{T,N}) where {T,N} = s
LinearAlgebra.adjoint(s::FormalSeries.Series{T,N}) where {T,N} = conj(s)
