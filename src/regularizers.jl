###############################################################################
### Regularizer - l1
###############################################################################
abstract type regularizer_l1 <: AbstractRegularizer end

g(pb::AbstractProblem{regularizer_l1}, x) = pb.λ * norm(x, 1)

softthresh(x, gamma) = sign(x) * max(0, abs(x)-gamma)
prox_αg(pb::AbstractProblem{regularizer_l1}, x, α) = softthresh.(x, α * pb.λ)

## regularization term for JuMP solve
function build_reg!(model, regtype::Type{regularizer_l1}, x, n)
    p = @variable model p[1:n]
    @constraint(model, [i = 1:n], 0 <= x[i]+p[i])
    @constraint(model, [i = 1:n], 0 <= -x[i]+p[i])
    return p
end

@inline function get_support(regtype::Type{regularizer_l1}, y; tol=1e-13)
    ind_null = SortedSet{Int}()
    for i in 1:length(y)
        abs(y[i])<tol && push!(ind_null, i)
    end
    return ind_null
end

###############################################################################
### Regularizer - TV
###############################################################################
abstract type regularizer_TV <: AbstractRegularizer end

g(pb::AbstractProblem{regularizer_TV}, x) = pb.λ* norm(x[1:end-1]-x[2:end], 1)

function prox_αg(pb::AbstractProblem{regularizer_TV}, x, α)
    res = zeros(length(x))
    TVdenoise!(res, x, α * pb.λ)
    return res
end

function build_reg!(model, regtype::Type{regularizer_TV}, x, n)
    p = @variable model p[1:n-1]
    @constraint(model, [i = 2:n], 0 <=  x[i]-x[i-1]+p[i-1])
    @constraint(model, [i = 2:n], 0 <= -x[i]+x[i-1]+p[i-1])
    return p
end

@inline function get_support(regtype::Type{regularizer_TV}, y; tol=1e-13)
    ind_null = SortedSet{Int}()
    for i in 1:length(y)-1
        abs(y[i] - y[i+1])<tol && push!(ind_null, i)
    end
    return ind_null
end

###############################################################################
### Regularizer - l1-2
###############################################################################
abstract type regularizer_l12{T} <: AbstractRegularizer end

function g(pb::AbstractProblem{regularizer_l12{T}}, x) where T
    res = 0.0

    n = size(x, 1)
    ngroups = Int(ceil(n/T))

    for i in 1:ngroups-1
        res += norm(x[T*i-3:T*i], 2)
    end
    res += norm(x[T*ngroups-3:end], 2)

    return pb.λ* res
end

function prox_αg(pb::AbstractProblem{regularizer_l12{T}}, x, α) where T
    res = copy(x)

    n = size(x, 1)
    ngroups = Int(ceil(n/T))

    for i in 1:ngroups-1
        x_gnorm = norm(x[T*i-3:T*i], 2)
        if x_gnorm >= α*pb.λ
            res[T*i-3:T*i] *= (x_gnorm - α*pb.λ) / x_gnorm
        else
            res[T*i-3:T*i] .= 0
        end
    end

    x_gnorm = norm(x[T*ngroups-3:end], 2)
    if x_gnorm >= α*pb.λ
        res[T*ngroups-3:end] *= (x_gnorm - α*pb.λ) / x_gnorm
    else
        res[T*ngroups-3:end] .= 0
    end

    return res
end

# function build_reg!(model, regtype::Type{regularizer_l12}, x, n)
#     p = @variable model p[1:n-1]
#     @constraint(model, [i = 2:n], 0 <=  x[i]-x[i-1]+p[i-1])
#     @constraint(model, [i = 2:n], 0 <= -x[i]+x[i-1]+p[i-1])
#     return p
# end

@inline function get_support(regtype::Type{regularizer_l12{T}}, y; tol=1e-13) where T
    ind_null = SortedSet{Int}()

    n = size(y, 1)
    ngroups = Int(ceil(n/T))

    for i in 1:ngroups-1
        norm(y[T*i-3:T*i], 2)<tol && push!(ind_null, i)
    end
    norm(y[T*ngroups-3:end], 2)<tol && push!(ind_null, ngroups)

    return ind_null
end

###############################################################################
### Regularizer - l_nuclear
###############################################################################
abstract type regularizer_lnuclear <: AbstractRegularizer end

function g(pb::AbstractProblem{regularizer_lnuclear}, x)
    n = Int(sqrt(size(x, 1)))
    x_mat = reshape(x, n, n)

    return pb.λ* norm(svdvals(x_mat), 1)
end

function prox_αg(pb::AbstractProblem{regularizer_lnuclear}, x, α)
    n = Int(sqrt(size(x, 1)))
    x_mat = reshape(x, n, n)

    F = svd(x_mat)
    res = vec(F.U * Diagonal(softthresh.(F.S, α*pb.λ)) * F.Vt)

    return res
end

# function build_reg!(model, regtype::Type{regularizer_lnuclear}, x, n)
#     p = @variable model p[1:n-1]
#     @constraint(model, [i = 2:n], 0 <=  x[i]-x[i-1]+p[i-1])
#     @constraint(model, [i = 2:n], 0 <= -x[i]+x[i-1]+p[i-1])
#     return p
# end

@inline function get_support(regtype::Type{regularizer_lnuclear}, y; tol=1e-13)
    n = Int(sqrt(size(y, 1)))
    y_mat = reshape(y, n, n)

    return SortedSet{Int}(1:n - rank(y_mat))
end

###############################################################################
### Regularizer - dist(⋅, B(0,1)) = max(||⋅||-1, 0)
###############################################################################
abstract type regularizer_distball{p} <: AbstractRegularizer end

g(pb::AbstractProblem{regularizer_distball{p}}, x) where p = pb.λ* max(norm(x, p)-1, 0)

function prox_αg(pb::AbstractProblem{regularizer_distball{p}}, x, α) where p
    xnorm = norm(x, p)
    if xnorm <= 1
        res = copy(x)
    elseif xnorm > 1+α*pb.λ
        res = x .* (1 - α*pb.λ / xnorm)
    else
        res = x ./ xnorm
    end
    return res
end

function build_reg!(model, regtype::Type{regularizer_distball{p}}, x, n) where p
    regval = @variable(model, regval)
    @constraint(model, 0 <= regval)
    @constraint(model, 0 <= sum(x[i]^2 for i in 1:n) - regval^2)
    return regval
end

@inline function get_support(regtype::Type{regularizer_distball{p}}, y; tol=1e-11) where p
    ind_null = SortedSet{Int}()
    abs(norm(y, p)-1) <= tol && push!(ind_null, 1)
    return ind_null
end
