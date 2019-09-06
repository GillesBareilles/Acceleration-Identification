using StructOpt
using DataStructures, LinearAlgebra, PGFPlotsX, Random

function get_problems()
    regularizers = Set([regularizer_l1, regularizer_TV])

    problems = Dict()

    nseeds = 5
    
    ## Lasso l1
    l1_pbs = []

    m, n, sparsity = 60, 128, 8
    Random.seed!(1234)
    xstart = rand(n) .* 10
    for seed in 1:nseeds
        pb = get_randomlasso(n, m, sparsity, reg=regularizer_l1, seed = seed)
        xorig = pb.x0
        
        push!(l1_pbs, (name = "pblasso_l1", pb = pb, xstart = xstart))
    end

    problems["l1_randinit"] = l1_pbs

    ## Lasso l1 zero init
    l1_zeroinit_pbs = []

    for pb in l1_pbs
        push!(l1_zeroinit_pbs, (name = "pblasso_l1", pb = pb.pb, xstart = zeros(n)))
    end

    problems["l1_zeroinit"] = l1_zeroinit_pbs

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
            :saveiter => false,
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
            :saveiter => false,
            # :αuser => αuser,
            :printstep => printstep,
            :itmax => itmax,
            :saveiter => false,
        ),
    ))
        
    push!(algorithms, (
        name="CI",
        updatefunc=extra_CondInertia,
        params=Dict(
            :saveiter => false,
            :id_testkind=>:default,
            # :αuser => αuser,
            :printstep => printstep,
            :itmax => itmax,
            :saveiter => false,
        ),
    ))
    
    push!(algorithms, (
        name="CIpred",
        updatefunc=extra_CondPredInertia,
        params=Dict(
            :saveiter => false,
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