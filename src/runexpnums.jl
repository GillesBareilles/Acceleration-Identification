export simplify_coords, normalize_identified_coords

function simplify_coords(coords)
    new_coords = [ coords[1] ]
    for coord_ind in 2:length(coords)-1
        if (coords[coord_ind][2] != coords[coord_ind+1][2]) || (coords[coord_ind][2] != coords[coord_ind-1][2])
            push!(new_coords, coords[coord_ind])
        end
    end
    push!(new_coords, coords[end])
    return new_coords
end

function normalize_identified_coords(coords_nbman_id, opt_supp)
    supp_size = length(opt_supp)

    return normalized_coords = [ (t[1], 100 * t[2]/supp_size) for t in coords_nbman_id]
end


function runexpnums(problems, algorithms; FIGS_FOLDER = "./figs/expnums")

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
    basename(pwd()) == "src" && (FIGS_FOLDER = joinpath("..", FIGS_FOLDER))

    for (pbclass_name, algo_to_stats) in problemclass_to_algstats
        println("Generating graphs for class ", pbclass_name)

        ps = []

        colors_vec = distinguishable_colors(8) # transform=deuteranopic

        algo_id = 1
        for (algo, stats) in algo_to_stats
            @show algo.name

            ############################################
            ## Plot one for legend
            (pb, pbinstance_stats) = first(stats); delete!(stats, pb)

            reg = typeof(pb.pb).parameters[1]
            opt_supp = get_support(reg, pb.pb.x0)

            coords_nbman_y_identified = simplify_coords([ (k, length(intersect(v, opt_supp))) for (k, v) in pbinstance_stats.hist[:iter_support_y]])
            coords_nbman_y_identified = normalize_identified_coords(coords_nbman_y_identified, opt_supp)

            push!(ps, PlotInc(
                PGFPlotsX.Options(
                    "no marks" => nothing,
                    # "line width" => "0.3pt",
                    # "color" => colors_vec[algo_id],
                    get_traj_params(algo.name)...
                ),
                Coordinates(coords_nbman_y_identified)
            ))

            for (pb, pbinstance_stats) in stats
                reg = typeof(pb.pb).parameters[1]
                opt_supp = get_support(reg, pb.pb.x0)

                coords_nbman_y_identified = simplify_coords([ (k, length(intersect(v, opt_supp))) for (k, v) in pbinstance_stats.hist[:iter_support_y]])
                coords_nbman_y_identified = normalize_identified_coords(coords_nbman_y_identified, opt_supp)

                push!(ps, PlotInc(
                    PGFPlotsX.Options(
                        "no marks" => nothing,
                        "forget plot" => nothing,
                        # "line width" => "0.3pt",
                        # "color" => colors_vec[algo_id],
                        get_traj_params(algo.name)...
                    ),
                    Coordinates(coords_nbman_y_identified)
                ))

            end
            push!(ps, LegendEntry(get_legend(algo.name)))

            algo_id+=2
        end

        fig = @pgf Axis(
            {
                # height = "12cm",
                # width = "12cm",
                # xlabel = "iter",
                # ylabel = L"F(x_k)-F^\star",
                # ymode  = "log",
                xmode  = "log",
                # title = "Problem $(pb.name) -- Suboptimality",
                legend_pos = "north west",
                legend_cell_align = "left",
                legend_style = "font=\\footnotesize",
            },
            ps...
        )
        try
            pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.pdf"), fig)
        catch e
            println("Error while building pdf:")
            println(e)
        end
        pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.tikz"), fig)
        ##################################
    end

    return
end
