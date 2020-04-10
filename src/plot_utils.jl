using PGFPlotsX, LaTeXStrings

include("plot_params.jl")

function add_identif_point!(ps, pb, stats, inds, vals; color="black")
    minval = pb.objopt

    k1, k2 = collect(keys(stats.hist[:iter_support_x]))
    logstep = k2-k1

    reg = typeof(pb.pb).parameters[1]
    opt_support = get_support(reg, pb.pb.x0)
    indices_identif = SortedSet(keys(filter(kv->collect(kv.second) == collect(opt_support), stats.hist[:iter_support_x])))

    if length(indices_identif) > 0
        itmax = maximum(keys(stats.hist[:iter_support_x]))
        itidentif = itmax + logstep
        while itidentif-logstep in indices_identif
            itidentif -= logstep
        end

        if itidentif <= itmax
            res = findall(x->x==itidentif, inds)# find it st inds[it] = itidentif, res = vals[it]
            indidentif = res[1]

            push!(ps, PlotInc(
                PGFPlotsX.Options(
                    "forget plot" => nothing,
                    "draw" => "none",
                    "mark" => "oplus",
                    "color" => color
                ),
                Coordinates([ (inds[indidentif], vals[indidentif]) ]),
            ))
        end
    end

    return
end

function build_save_suboptimality(pb, algo_to_stats, FIGS_FOLDER)
    minval = pb.objopt

    ps = []
    algo_id = 1
    for (algo, stats) in algo_to_stats
        inds = collect(keys(stats.hist[:iter_f]))
        vals = collect(abs.(values(stats.hist[:iter_f]) .- minval))
        coords = [ (inds[i], vals[i]) for i in 1:length(vals)]

        push!(ps, PlotInc(
            PGFPlotsX.Options(
                get_suboptimality_algoparams(algo.name)...
            ),
            Coordinates(coords)
        ))
        push!(ps, LegendEntry(get_legend(algo.name)))

        ## Add identification point
        add_identif_point!(ps, pb, stats, inds, vals, color=get_alg_color(algo.name))

        algo_id+=1
    end

    fig = @pgf Axis(
        {
            # height = "12cm",
            # width = "12cm",
            xlabel = "number of proximal gradient steps",
            legend_cell_align = "left",
            xmin = 0,
            ylabel = L"F(x_k)-F^\star",
            ymode  = "log",
            legend_style = "{font=\\footnotesize}",
            # xmode  = "log",
            # title = "Problem $(pb.name) -- Suboptimality",
            legend_pos = "north east",
        },
        ps...
    )
    pgfsave(joinpath(FIGS_FOLDER, "$(pb.name)_subopt.pdf"), fig)
    pgfsave(joinpath(FIGS_FOLDER, "$(pb.name)_subopt.tikz"), fig)
end


###############################################################################
#### Iterates figure functions

function add_contour!(ps, f, x, y)
    push!(ps, PlotInc(
            PGFPlotsX.Options(
                "forget plot" => nothing,
                "no marks" => nothing,
                "ultra thin" => nothing
            ),
            Table(contours(x, y, f.(x, y'), 10)),
    ))
    return
end

function add_traj!(ps, coords, algoname; params=Dict())
    if length(coords)>0
        push!(ps, PlotInc(
            PGFPlotsX.Options(params...),
            Coordinates(coords),
        ))
        push!(ps, LegendEntry(get_legend(algoname)))
    end
    return
end

function add_point!(ps, xopt)
    coords = [ (xopt[1], xopt[2]) ]

    push!(ps, PlotInc(
        PGFPlotsX.Options(
            "forget plot" => nothing,
            "only marks" => nothing,
            "mark" => "star",
            "thick" => nothing,
            "color" => "black"
            ),
        Coordinates(coords),
    ))

    return
end

function pgf_build_iteratesfig(ps, xmin, xmax, ymin, ymax)
    return @pgf Axis(
        {
            contour_prepared,
            # height = "12cm",
            # width = "12cm",
            # xlabel = "x",
            # ylabel = "y",
            xmin = xmin,
            xmax = xmax,
            ymin = ymin,
            ymax = ymax,
            legend_pos = "north east",
            legend_cell_align = "left",
            legend_style = "font=\\footnotesize",
            # title = "Problem $(pb.name) -- Iterates postion",
        },
        ps...,
    )
end

function add_manifold!(ps, coords)
    push!(ps, PlotInc(
        PGFPlotsX.Options(
            "forget plot" => nothing,
            "no marks" => nothing,
            "smooth" => nothing,
            "thick" => nothing,
            "solid" => nothing,
            "black!50!white" => nothing,
            # "mark size" => "1pt"
        ),
        Coordinates(coords),
    ))
    return
end
