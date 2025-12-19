
#############
# Save data #
#############

"""
Given an array of formalseries, get an array of the same size with the selected
order
"""
function get_sorder(arr::Array{Series{T,N}}, order) where {T,N}
    A = zeros(T, size(arr))
    for i in eachindex(arr)
        A[i] = arr[i].c[order]
    end
    return A
end

"""
Given an array of formalseries, returns an array with an aditional dimension (in
the last axis) where each index in it corresponds to the orders 1:N of the
formalseries
"""
function series_stack(arr::Array{Series{T,N}}) where {T,N}
    A = zeros(T, (size(arr)...,N)) 
    for i in 1:N
        A[..,i] .= get_sorder(arr, i)
    end
    return A
end

"""
Given an N-dimensional array where the last axis contains the orders 1:N, it
returns a formalseries array from it (with one dimension less).
"""
function series_unstack(arr::Array{T}) where T
    sz = size(arr)
    N = sz[end]
    A = zeros(Series{T,N}, sz[1:end-1])
    for i in CartesianIndices(A)
        A[i] = Series{T,N}(ntuple(j -> arr[i,j], N))
    end
    return A
end
