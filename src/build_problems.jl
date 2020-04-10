###############################################################################
## LASSO Problem
###############################################################################
function get_randomlasso(n, m, sparsity; reg=regularizer_l1, seed=1234)
    # A is drawn from the standard normal distribution
    # b = Ax0 + e where e is taken from the normal distribution with standard deviation 0.001.
    # We set λ1 so that the original sparsity is ultimately recovered.

    Random.seed!(seed)
    A = rand(Normal(), m, n)

    Random.seed!(seed)
    x0 = rand(Normal(), n)

    if reg == regularizer_l1
        Random.seed!(seed)
        inds_nz = Set(randperm(n)[1:sparsity])

        for i in 1:n
            !(i in inds_nz) && (x0[i] = 0)
        end

    elseif reg <: regularizer_l12
        T = reg.parameters[1]
        ngroups = Int(ceil(n/T))

        Random.seed!(seed)
        inds_nz = Set(randperm(ngroups)[1:sparsity])

        for i in 1:ngroups-1
            !(i in inds_nz) && (x0[T*i-3:T*i] .= 0)
        end
        !(ngroups in inds_nz) && (x0[T*ngroups-3:end] .= 0)

    elseif reg == regularizer_lnuclear
        n_mat = Int(sqrt(n))

        Random.seed!(seed)
        basisvecs = rand(Normal(), n_mat, sparsity)

        x0 = zeros(n)
        for sp in 1:sparsity
            x0 += vec(basisvecs[:, sp] * transpose(basisvecs[:, sp]))
        end

    elseif reg <: regularizer_distball
        p = reg.parameters[1]

        Random.seed!(seed)
        x0 = rand(Normal(), n); x0 /= norm(x0, p)

    else
        @error "hunhandled regularizer $reg"
    end

    delta = 0.01
    Random.seed!(seed)
    e = rand(Normal(0, delta^2), m)

    y = A*x0+e
    pb = LassoPb{reg}(A, y, delta, n, x0)

    return pb
end

function get_IM_testcase(; n=80, m=130, sparsity=0.1)
    # A is drawn from the standard normal distribution
    # b = Ax0 + e where x0 is a 10% sparse vector taken from the normal distribution,
    # e is taken from the normal distribution with standard deviation 0.001.
    # We set λ1 so that the original sparsity is ultimately recovered.

    Random.seed!(23)
    A = rand(Normal(), m, n)

    Random.seed!(23)
    x0 = rand(Normal(), n)
    for i in 1:n
        rand() < sparsity && (x0[i]=0)
    end

    Random.seed!(23)
    e = rand(Normal(0, 0.001), m)

    y = A*x0+e

    return LassoPb{regularizer_l1}(A, y, 1., n, x0)
end

function get_randomlassopb_TV(; n, p, λ=1, TVsparsity=0.05)
    Random.seed!(23)
    A = rand(Float64, p, n)


	y_orig = zeros(n)
	y = zeros(n)
	y_orig[1] = 10

	## Ground truth vector
	Random.seed!(23); sparsity = floor.(rand(n-1) .+ TVsparsity)
	Random.seed!(23);
	aux = rand(Normal(0, 4.0), n-1) .* sparsity
	for i in 2:n
		y_orig[i] = y_orig[i-1] + aux[i-1]
	end

	## Noisy observation
	Random.seed!(23); pert = rand(Normal(0, 1.0), n)
    y = y_orig .+ pert

    y_obs = A * y

    return LassoPb{regularizer_TV}(A, y_obs, λ, n, y_orig)
end

function build_2dlasso_normdsol(;p=2, θ = π*8/(9*2))
    xcenter = [cos(θ), sin(θ)]
    xcenter /= norm(xcenter, p)

    diagscaling = [1, 10]
    A = Diagonal(diagscaling)
    y = diagscaling .* xcenter
    λ = 1

    return LassoPb{regularizer_distball{p}}(A, y, λ, 2, xcenter)
end

###############################################################################
## Logistic Problem
###############################################################################

## ionosphere problem
function get_ionopb()
    @assert isfile("data/ionosphere.data")

    rawdata = readdlm("data/ionosphere.data", ',')
    A = Matrix{Float64}(rawdata[:, 1:end-1])
    n = size(A, 2)

    y = map(x-> x=="g" ? 1 : -1, rawdata[:, end])

    return LogisticPb{regularizer_l1}(A, y, 0.1, n, zeros(1))
end

## Random sparse logit
function get_logit(;n=80, m=85, sparsity=0.5)

    A = rand(m, n)*10
    y = Vector([rand()>sparsity for i in 1:m])

    return LogisticPb{regularizer_l1}(A, y, 0.1, n, zeros(1))
end