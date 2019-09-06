module StructOpt

using Printf, DataStructures

using LinearAlgebra, DelimitedFiles, Random, Distributions
using JuMP, Ipopt


## For plotting
using PGFPlotsX, LaTeXStrings, Colors, Contour


export f, g, F, ∇f, prox_αg, get_gradlips, get_μ_cvx, get_optimsol, get_support

export solve_proxgrad, extra_ISTA, extra_FISTA, extra_CondInertia, extra_CondPredInertia, extra_altin_cond, extra_ADAPTsubspace, extra_FISTArestart, extra_FISTArestart_cond, extra_MFISTA, extra_AltIntertia

export LassoPb, LogisticPb
export regularizer_l1, regularizer_l12, regularizer_TV, regularizer_lnuclear, regularizer_distball


export get_randomlasso, get_IM_testcase, get_randomlassopb_TV, build_2dlasso_normdsol, get_ionopb, get_logit

export make_binbounds, make_bins, add_identif_point!, build_save_suboptimality, build_save_steplength, add_contour!, add_traj!, add_point!, pgf_build_iteratesfig, add_manifold!, get_alg_color, get_traj_params, get_legend, runexpnums

include("TV_solver.jl")

###############################################################################
### Problem abstract structure

abstract type AbstractRegularizer end
abstract type AbstractProblem{T} end

# function
f(pb::AbstractProblem, x) = error("NotImplementedError: f for type $(typeof(pb)).")
g(pb::AbstractProblem{T}, x) where T = error("NotImplementedError: g for problem type $T.")

F(pb::AbstractProblem{T}, x) where {T <: AbstractRegularizer} = f(pb, x) + g(pb, x)

# 1st order
∇f(pb::AbstractProblem, x) = error("NotImplementedError: ∇f for type $(typeof(pb)).")
prox_αg(pb::AbstractProblem{T}, x, α) where {T<:AbstractRegularizer} = error("NotImplementedError: prox_αg for problem type $T.")

# conditioning
get_gradlips(pb::AbstractProblem) = error("NotImplementedError: get_gradlips for type $(typeof(pb)).")
get_μ_cvx(pb::AbstractProblem) = -Inf


###############################################################################
## LASSO Problem parametrized by regularizer reg
#    min_x  0.5 * ||Ax-y||² + λ reg(x)

mutable struct LassoPb{T} <: AbstractProblem{T}
    A::Matrix{Float64}
    y::Vector{Float64}
    λ::Float64
    n::Int
    x0::Vector{Float64}
end

get_gradlips(pb::LassoPb) = opnorm(pb.A)^2
get_μ_cvx(pb::LassoPb) = (svdvals(pb.A)[end])^2

f(pb::LassoPb, x) = 0.5 * norm(pb.A*x-pb.y)^2
∇f(pb::LassoPb, x) = transpose(pb.A)*(pb.A*x - pb.y)


function get_optimsol(pb::LassoPb{T}) where {T <: AbstractRegularizer}
    model = Model(with_optimizer(Ipopt.Optimizer, tol=1e-13, print_level=0))
    n = pb.n
    
    t = @variable model t
    x = @variable model x[1:n]

    p = build_reg!(model, T, x, n)
    
    @objective model Min 0.5 * transpose(pb.A*x-pb.y) * (pb.A*x-pb.y) + pb.λ * t
    @constraint model t >= sum(p)
    
    optimize!(model)

    objopt = objective_value(model)
    xopt = value.(x)

    return objopt, xopt
end


###############################################################################
## Logistic Problem
#    min_x  1/m * ∑_{i=1}^m log(1 + exp(-y_i dot(A_i, x))) + λ reg(x)
#
#    x : R^n
#    y : R^m (observations)
#    A : mxn matrix of samples

mutable struct LogisticPb{T} <: AbstractProblem{T}
    A::Matrix{Float64}
    y::Vector{Float64}
    λ::Float64
    n::Int
    x0::Vector{Float64}
end

function f(pb::LogisticPb, x)
    m = size(pb.A, 1)
    
    return (1/m) * sum(log(1 + exp(-pb.y[i] * dot(pb.A[i, :], x))) for i in 1:m)
end

σ(x) = 1/(1+exp(-x))
function ∇f(pb::LogisticPb, x)
    m = size(pb.A, 1)
   
    vec = zeros(m)
    for i in 1:m
        vec[i] = (1/m) * σ(-pb.y[i] * dot(pb.A[i, :], x)) * (-pb.y[i])
    end
    return transpose(pb.A) * vec
end


function get_gradlips(pb::LogisticPb)
    m = size(pb.A, 1)
    return opnorm(pb.A)^2 / m
end


function get_optimsol(pb::LogisticPb)
    model = Model(with_optimizer(Ipopt.Optimizer, tol=1e-13, print_level=0))
    n = pb.n
    m = size(pb.A, 1)
    
    t = @variable model t
    p = @variable model p[1:n] >= 0
    x = @variable model x[1:n]
    
    ## Linear part of objective...
    lins = @variable model lins[1:m]
    
    @NLobjective model Min (1/m) * sum(log(1 + exp(-pb.y[i] * lins[i])) for i in 1:m) + pb.λ * t
    
    my_expr = pb.A * x
    @constraint(model, [i=1:m], lins[i] == dot(pb.A[i, :], x))

    ## l1 norm modeling
    @constraint model t >= sum(p)
    @constraint(model, [i = 1:n], 0 <= x[i]+p[i])
    @constraint(model, [i = 1:n], 0 <= -x[i]+p[i])

    optimize!(model)

    objopt = objective_value(model)
    xopt = value.(x)

    return F(pb, xopt), xopt
end

include("alg_proxgrad.jl")
include("algs_extrapolation.jl")
include("regularizers.jl")
include("build_problems.jl")

include("runexpnums.jl")
include("plot_utils.jl")

end # module
