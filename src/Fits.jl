
"""
Given a function f, the x-data and the errors dy, returns

- function lmfit(prm, y), that given the parameters prm of the function f and
  the y-data returns the vector (y[i]-f(x[i],prm))/dy[i]

- function chisq(prm, y), that given prm and y returns the χ² value
"""
function fit_defs_yerr(f, x, dy)
    function lmfit(prm, y)
        nof = round(Int64, length(y))
        res = Vector{eltype(prm)}(undef,nof)
        for i in 1:nof
            res[i] = (y[i] - f(x[i], prm)) / dy[i]
        end
        return res
    end
    chisq(prm,data) = sum(lmfit(prm, data) .^ 2)
    return lmfit, chisq
end

"""
Same as fit_defs_yerr, but for a set of functions fs=[f1, f2, ...], x-data
xs=[x1, x2,...], errors dys=[dy1, dy2, ...].
"""
function fit_defs_yerr(fs::Vector{<:Function}, xs, dys)
    nofs = length(fs)
    nof = [length(xs[i]) for i in 1:nofs]
    idxs = [collect( (i == 1 ? 1 : sum(nof[1:i-1])+1):sum(nof[1:i])) for i in 1:nofs]
    function lmfit(prm, y)
        res = vcat(ntuple(i -> fit_defs_yerr(fs[i], xs[i], dys[i])[1](prm, y[idxs[i]]), nofs)...)
        return res 
    end
    chisq(prm,data) = sum(lmfit(prm, data) .^ 2)
    return lmfit, chisq
end

"""
Same as fit_defs_yerr for correlated fits, where instead of the errors dy one
inputs the inverse of the covariance matrix Winv. Should be the same as
fit_defs_yerr if Winv = diag(1/dy^2)
"""
function fit_defs_yerr_corr(f, x, Winv)
    u = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Winv)).U
    function lmfit(prm, y)
        nof  = round(Int64, length(y))
        res  = Vector{eltype(prm)}(undef,nof)
        res2 = Vector{eltype(prm)}(undef,nof)
        res =   (y .- f(x, prm))
        for k in 1:length(res)
            res2[k] = u[k,1]*res[1]
            for i in 2:length(res)
                res2[k] = res2[k] + u[k,i]*res[i]
            end
        end
        return res2
    end
    chisq(prm,data) = sum(lmfit(prm, data) .^ 2)
    return lmfit, chisq
end

"""
Same as fit_defs_yerr_corr, but for a set of functions fs=[f1,f2,...] and x-data
xs=[x1,x2,...]
"""
function fit_defs_yerr_corr(fs::Vector{<:Function}, xs, Winv)
    u = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Winv)).U
    nofs = length(fs)
    nof = [length(xs[i]) for i in 1:nofs]
    idxs = [collect( (i == 1 ? 1 : sum(nof[1:i-1])+1):sum(nof[1:i])) for i in 1:nofs]
    function lmfit(prm, y)
        res  = Vector{eltype(prm)}(undef,sum(nof))
        res2 = Vector{eltype(prm)}(undef,sum(nof))
        for i in 1:nofs
            res[idxs[i]] .=   (y[idxs[i]] .- fs[i].(xs[i], [prm]))
        end
        for k in 1:length(res)
            res2[k] = u[k,1]*res[1]
            for i in 2:length(res)
                res2[k] = res2[k] + u[k,i]*res[i]
            end
        end
        return res2
    end
    chisq(prm,data) = sum(lmfit(prm, data) .^ 2)
    return lmfit, chisq
end

"""
Construct covariance matrix of a uwreal vector uwv with id ID.
"""
function construct_cov(uwv, ID)
    N = length(uwv)
    M = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        M[i,j] = Statistics.cov(mchist(uwv[i], ID), mchist(uwv[j], ID))
    end
    return M
end



"""
    fit_routine(f::Function, xdat, ydat, npar)

Given a function f, x-xdata xdat, (uwreal) y-data ydat, and the numer of
parameters of the function f, returns the fit parameters
"""
function fit_routine(f::Function, xdat, ydat, npar)
    prms0 = fill(0.5, npar)
    uwerr.(ydat)
    lm, csq = fit_defs_yerr(f, xdat, ADerrors.err.(ydat))
    fit  = LeastSquaresOptim.optimize(xx -> lm(xx, value.(ydat)), prms0, LeastSquaresOptim.LevenbergMarquardt(), autodiff = :forward)
    fitp, csqexp = ADerrors.fit_error(csq, fit.minimizer, ydat)
    csqv = csq(value.(fitp), value.(ydat))
    csqr = csqv / csqexp
    println("χ = $(csqv)")
    println("χexp = $(csqexp)")
    println("χr = $(csqr)")
    return fitp
end

"""
    fit_routine(fs::Vector{Function}, xdat, ydat, npar; ENS, correlated = false)

Same as above but for a set of functions functions fs=[f1,f2,...],
x-data xdat=[xdat1,xdat2,...], y-data ydat=[ydat1,ydat2,...]. The parameters of
the functions fi must be shared among all the functions fi. Needs the uwreal ID
in ENS.
"""
function fit_routine(fs::Vector{Function}, xdat, ydat, npar; ENS, correlated = false)
    prms0 = fill(0.5, npar)
    nofs = length(fs)
    for i in 1:nofs
        uwerr.(ydat[i])
    end
    yvals = vcat(ntuple(i -> ydat[i], nofs)...)
    if correlated == false
        errs = [ADerrors.err.(item) for item in ydat]
        lm, csq = fit_defs_yerr(fs, xdat, errs)
        fit  = LeastSquaresOptim.optimize(xx -> lm(xx, value.(yvals)), prms0, LeastSquaresOptim.LevenbergMarquardt(), autodiff = :forward)
        fitp, csqexp = ADerrors.fit_error(csq, fit.minimizer, yvals)
    elseif correlated == true
        cvi = LinearAlgebra.inv(construct_cov(yvals, ENS)) # correlated
        # cvi = diagm(1 ./ADerrors.err.(yfit).^2) # not correlated
        lm, csq = fit_defs_yerr_corr(fs, xdat, cvi)
        fit  = LeastSquaresOptim.optimize(xx -> lm(xx, value.(yvals)), prms0, LeastSquaresOptim.LevenbergMarquardt(), autodiff = :forward)
        fitp, csqexp = ADerrors.fit_error(csq, fit.minimizer, yvals, W=cvi)
    end
    csqv = csq(value.(fitp), value.(yvals))
    csqr = csqv / csqexp
    println("χ = $(csqv)")
    println("χexp = $(csqexp)")
    println("χr = $(csqr)")
    return fitp
end
