using StructOpt
using DataStructures, LinearAlgebra, PGFPlotsX, Random, Distributions, Colors

function get_problems()
    problems = Dict()

    nseeds = 1
    
    ## Lasso nuclear norm
    lnuclear_pbs = []

    # m, n, sparsity = 8^2, 10^2, 6
    m, n, sparsity = 16^2, 20^2, 3
    Random.seed!(1234)
    xstart = rand(Normal(), n) .* 10
    for seed in 1:nseeds
        pb = get_randomlasso(n, m, sparsity, reg=regularizer_lnuclear, seed = seed)
        xorig = pb.x0

        @show get_support(regularizer_lnuclear, xorig)
        @show get_support(regularizer_lnuclear, xstart)

        println("Computing objopt")
        xstar, hist = solve_proxgrad(pb, xstart, extra_FISTA; itmax=1e5, tol=1e-13, saveiter=false)

        objopt = minimum(values(hist[:iter_f]))
        @show objopt
        @show F(pb, xstar)

        push!(lnuclear_pbs, (name = "pblasso_lnuclear", pb = pb, xstart = xstart, objopt = objopt))
    end

    problems["lnuclear_randinit"] = lnuclear_pbs
 
    ## Lasso nuclear norm - zero init
    lnuclear_zeroinit_pbs = []
    for pb in lnuclear_pbs
        push!(lnuclear_zeroinit_pbs, (name = "pblasso_lnuclear", pb = pb.pb, xstart = zeros(n), objopt = pb.objopt))
    end
    
    problems["lnuclear_zeroinit"] = lnuclear_zeroinit_pbs

    return problems
end

function get_algorithms()
    algorithms = []
    
    # αuser = 0.001
    itmax = 3e4
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


function main()

    problems = get_problems()
    algorithms = get_algorithms()
    FIGS_FOLDER="./temp"

    ## Start from interesting point and optimize with ISTA, FISTA.
    problemclass_to_algstats = OrderedDict()
    for (problems_class_name, problem_class) in problems
        printstyled("\n---- Class $(problems_class_name)\n", color=:light_blue)

        algo_to_stats = OrderedDict()
        for (algo_id, algo) in enumerate(algorithms)
            printstyled("\n---- Algorithm $(algo.name)\n", color=:light_blue)
            
            algo_to_stats[algo] = Dict()
            for (pb_id, pb_data) in enumerate(problem_class)
                printstyled("\n----- Solving problem $pb_id, $(pb_data.name)\n", color=:light_blue)
                
                xopt, hist = solve_proxgrad(pb_data.pb, pb_data.xstart, algo.updatefunc; algo.params...)
                algo_to_stats[algo][pb_data] = (hist=hist, xopt=xopt)
            end
        end

        problemclass_to_algstats[problems_class_name] = algo_to_stats
    end


    ####################################################
    ## Display suboptimality, iterates position.
    FIGS_FOLDER = "./figs"
    basename(pwd()) == "src" && (FIGS_FOLDER = joinpath("..", FIGS_FOLDER))
    !ispath(FIGS_FOLDER) && mkpath(FIGS_FOLDER)


    alg_to_fig = Dict()

    for (pbclass_name, algo_to_stats) in problemclass_to_algstats
        println("Generating graphs for class ", pbclass_name)
        
        colors_vec = distinguishable_colors(8) # transform=deuteranopic

        # algo_id = 1
        for (algo, stats) in algo_to_stats
            ps_nidentif, ps_subopt = [], []
            @show algo.name

            ############################################
            ## N identified manifolds - 1st problem
            (pb, pbinstance_stats) = first(stats); delete!(stats, pb)
            
            reg = typeof(pb.pb).parameters[1]
            opt_supp = get_support(reg, pb.pb.x0)

            coords_nbman_y_identified = simplify_coords([ (k, length(intersect(v, opt_supp))) for (k, v) in pbinstance_stats.hist[:iter_support_y]])
            coords_nbman_y_identified = normalize_identified_coords(coords_nbman_y_identified, opt_supp)

            push!(ps_nidentif, PlotInc(
                PGFPlotsX.Options(
                    "no marks" => nothing,
                    # "line width" => "0.3pt",
                    # "color" => colors_vec[algo_id],
                    get_traj_params(algo.name)...
                ),
                Coordinates(coords_nbman_y_identified)
            ))

            ############################################
            ## subopt - 1st problem
            minval = pb.objopt

            inds = collect(keys(pbinstance_stats.hist[:iter_f]))
            vals = collect(abs.(values(pbinstance_stats.hist[:iter_f]) .- minval))
            coords = [ (inds[i], vals[i]) for i in 1:length(vals) if mod(i, 10) == 0]
        
            push!(ps_subopt, PlotInc(
                PGFPlotsX.Options(
                    "mark" => "none", 
                    "color" => "black",
                ),
                Coordinates(coords)
            ))
            add_identif_point!(ps_subopt, pb, pbinstance_stats, inds, vals, color=get_alg_color(algo.name))

            push!(ps_nidentif, LegendEntry(get_legend(algo.name)))
            
            fig = @pgf TikzPicture(
                Axis(
                    {
                        "scale only axis",
                        axis_y_line => "left",
                        ymin=0,
                        ymax=100,
                        xmajorgrids,
                        ymajorgrids,
                        yminorgrids,
                        # ylabel_style = "align=center",
                        # ylabel="Plot 1",
                        legend_pos = "north east",
                        legend_cell_align = "left",
                        legend_style = "font=\\footnotesize",
                    },
                    ps_nidentif...
                ),
                Axis(
                    {
                        ymin=1e-9,
                        ymax=1e1,
                        ymode  = "log",
                        # xmode  = "log",
                        "scale only axis",
                        axis_y_line => "right",
                        axis_x_line = "none",
                        # ylabel_style = "align=center",
                        # ylabel = "Plot 2",
                    },
                    ps_subopt...
                ),
            )

            pgfsave(joinpath(FIGS_FOLDER, "$(pbclass_name)_$(algo.name).pdf"), fig)
            pgfsave(joinpath(FIGS_FOLDER, "$(pbclass_name)_$(algo.name).tikz"), fig)
        end
    end
    

    return
end

main()