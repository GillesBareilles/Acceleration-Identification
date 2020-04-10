using StructOpt
using DataStructures, LinearAlgebra, PGFPlotsX, Random

function get_problems()
    regularizers = Set([regularizer_l1, regularizer_TV])

    problems = Dict()

    nseeds = 5

    ## Lasso l12
    l12_pbs = []

    m, n, sparsity = 60, 128, 3
    Random.seed!(1234)
    xstart = rand(n) .* 10
    for seed in 1:nseeds
        pb = get_randomlasso(n, m, sparsity, reg=regularizer_l12{4}, seed = seed)
        xorig = pb.x0

        push!(l12_pbs, (name = "pblasso_l12", pb = pb, xstart = xstart))
    end

    problems["l12_randinit"] = l12_pbs

    ## Lasso l12 zero init
    l12_zeroinit_pbs = []

    for pb in l12_pbs
        push!(l12_zeroinit_pbs, (name = "pblasso_l12", pb = pb.pb, xstart = zeros(n)))
    end

    problems["l12_zeroinit"] = l12_zeroinit_pbs

    return problems
end

function get_algorithms()
    algorithms = []

    # αuser = 0.001
    itmax = 4e4
    printstep = 1e3
    push!(algorithms, (
        name="FISTA",
        updatefunc=extra_FISTA,
        params=Dict(
            # :αuser => αuser,
            :printstep => printstep,
            :itmax => itmax,
            :saveiter => false,
        ),
    ))

    push!(algorithms, (
        name="ISTA",
        updatefunc=extra_ISTA,
        params=Dict(
            # :αuser => αuser,
            :printstep => printstep,
            :itmax => itmax,
            :saveiter => false,
        ),
    ))

    push!(algorithms, (
        name="T1",
        updatefunc=extra_CondInertia,
        params=Dict(
            :id_testkind=>:default,
            # :αuser => αuser,
            :printstep => printstep,
            :itmax => itmax,
            :saveiter => false,
        ),
    ))

    push!(algorithms, (
        name="T2",
        updatefunc=extra_CondPredInertia,
        params=Dict(
            # :αuser => αuser,
            :printstep => printstep,
            :itmax => itmax,
            :saveiter => false,
        ),
    ))

    return algorithms
end

FIGS_FOLDER = "./figs"
basename(pwd()) == "src" && (FIGS_FOLDER = joinpath("..", FIGS_FOLDER))
!ispath(FIGS_FOLDER) && mkpath(FIGS_FOLDER)

runexpnums(get_problems(), get_algorithms(), FIGS_FOLDER=FIGS_FOLDER)