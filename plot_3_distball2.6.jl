using StructOpt
using DataStructures, LinearAlgebra, PGFPlotsX

function main()
    ## build Lasso pball reg problem, with solution on partly smooth manifold ||x||=1, dimension 2.
    problems = []
    
    ## Distance to p norm unit ball
    θ = π*(0.5 + 0.2)
    xstart = 3 .* [cos(θ), sin(θ)]
    xstart = [1.5, 0.5]

    p = 2.6

    xsol = [cos(θ), sin(θ)]; xsol /= norm(xsol, p)

    A = Matrix{Float64}([1. 1.; 0. 1.])
    y = A * xsol
    λ = 1

    pb = LassoPb{regularizer_distball{p}}(A, y, λ, 2, xsol)
    objopt, xopt = get_optimsol(pb)
    objopt = 0

    push!(problems, (
        name = "distunitball-norm$p",
        pb = pb,
        x0 = xstart,
        xopt = xopt,
        objopt = objopt,
    ))


    ## Build algorithms
    algorithms = []
    
    αuser = 0.05
    # αuser = 0.1
    
    push!(algorithms, (
        name="ISTA",
        updatefunc=extra_ISTA,
        params=Dict(
            :αuser => αuser,
            :printstep => 1,
        ),
        # params=Dict(),
    ))

    push!(algorithms, (
        name="FISTA",
        updatefunc=extra_FISTA,
        # params=Dict(:αuser=>αuser),
        params=Dict(
            :αuser => αuser,
            :printstep => 1,
        ),
    ))
        
    push!(algorithms, (
        name="T1",
        updatefunc=extra_CondInertia,
        params=Dict(
            :id_testkind=>:default,
            :αuser => αuser,
            :printstep => 1,
        ),
    ))
    
    push!(algorithms, (
        name="T2",
        updatefunc=extra_CondPredInertia,
        params=Dict(
            :printstep => 1,
            :αuser => αuser,
            :proxgrad_per_it => 2,
        ),
    ))


    ## Start from interesting point and optimize with ISTA, FISTA.
    problem_to_algstats = OrderedDict()
    for (pb_id, pb_data) in enumerate(problems)
        printstyled("\n----- Solving problem $pb_id, $(pb_data.name)\n", color=:light_blue)
        
        algo_to_stats = OrderedDict()
        for (algo_id, algo) in enumerate(algorithms)
            printstyled("\n---- Algorithm $algo_id, $(algo.name)\n", color=:light_blue)
            
            xopt, hist = solve_proxgrad(pb_data.pb, pb_data.x0, algo.updatefunc; algo.params...)
            algo_to_stats[algo] = (hist=hist, xopt=xopt)
        end
        
        problem_to_algstats[pb_data] = algo_to_stats
    end


    ####################################################
    ## Display suboptimality, iterates position.
    FIGS_FOLDER = "./figs"
    basename(pwd()) == "src" && (FIGS_FOLDER = joinpath("..", FIGS_FOLDER))
    !ispath(FIGS_FOLDER) && mkpath(FIGS_FOLDER)

    for (pb, algo_to_stats) in problem_to_algstats
        println("Generating graphs for ", pb.name)
        
        ## Suboptimality plot
        build_save_suboptimality(pb, algo_to_stats, FIGS_FOLDER)
        
        ## Steplength plot
        # build_save_steplength(pb, algo_to_stats, FIGS_FOLDER)
        
        if pb.pb.n == 2
            reg = typeof(pb.pb).parameters[1]
            
            xsol = pb.pb.x0
            
            δ = 2e-3
            xmin, xmax = xsol[1]-δ, xsol[1]+δ
            ymin, ymax = xsol[2]-δ, xsol[2]+δ
            
            ## Iterates position
            ps = []
            
            ## Plot contour
            x = xmin:(xmax - xmin) / 100:xmax
            y = ymin:(ymax - ymin) / 100:ymax
            f = (x,y) -> 0.5 * norm(pb.pb.A * [x, y] - pb.pb.y)^2
            add_contour!(ps, f, x, y)

            for (algo, stats) in algo_to_stats
                coords = [ (point[1], point[2]) for (iter, point) in stats.hist[:iter_x]  if xmin < point[1] < xmax && ymin < point[2] < ymax]
        
                add_traj!(ps, coords, algo.name, params=get_traj_params(algo.name))
            end

            ## Plot optimal point
            add_point!(ps, xopt)

            # plot optimal manifold
            @assert reg <: regularizer_distball
            d = reg.parameters[1]

            θs = -π:0.0001:π
            xs, ys = cos.(θs), sin.(θs)
            coords = []
            for i in 1:length(xs)
                ptnorm = norm([xs[i] ys[i]], d)
                if xmin <= xs[i]/ptnorm <= xmax && ymin <= ys[i]/ptnorm <= ymax
                    push!(coords, (xs[i]/ptnorm, ys[i]/ptnorm))
                end
            end
            length(coords)>0 && add_manifold!(ps, coords)

            fig = pgf_build_iteratesfig(ps, xmin, xmax, ymin, ymax)

            pgfsave(joinpath(FIGS_FOLDER, "$(pb.name)_iterates.pdf"), fig)
            pgfsave(joinpath(FIGS_FOLDER, "$(pb.name)_iterates.tikz"), fig)
        end
    end

    println("\n")
    for (pb, algo_to_stats) in problem_to_algstats
        @show pb.name
        for (algo, stats) in algo_to_stats
            @show algo.name
            display(stats[:xopt])
            # @show keys(stats)

        end

    end

    return
end

main()