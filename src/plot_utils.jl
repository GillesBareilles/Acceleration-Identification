using PGFPlotsX, LaTeXStrings

function make_binbounds(maxkey, nbins)
    binbounds = []
    
    binsize = floor((maxkey-2)/(nbins-2))
    
    for i in 0:nbins-3
        lowbin = binsize*(i)+2
        upbin = binsize*(i+1)+1

        push!(binbounds, (lowbin, upbin))
    end

    lowbin = binsize*(nbins-2)+2
    upbin = maxkey
    push!(binbounds, (lowbin, upbin))
    return binbounds
end

function make_bins(data, maxkey; nbins=20)
    binbounds = make_binbounds(maxkey, nbins)
    
    t = [
        ("1", get(data, 1, 0))
    ]
    
    for (lowbin, upbin) in binbounds
        count = 0
        for jumpval in lowbin:upbin
            count += get(data, jumpval, 0)
        end
        
        push!(t, ("$(Int(lowbin))-$(Int(upbin))", count))
    end
    
    return t
end

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
                "mark" => "none", 
                "color" => get_alg_color(algo.name),
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
            xlabel = "proxgrad calls",
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

function build_save_steplength(pb, algo_to_stats, FIGS_FOLDER)
    ps = []

    algo_id = 1
    for (algo, stats) in algo_to_stats
        inds = collect(keys(stats.hist[:iter_step_length]))
        vals = collect(values(stats.hist[:iter_step_length]))
        coords = [ (inds[i], vals[i]) for i in 1:length(vals)]

        push!(ps, PlotInc(
            PGFPlotsX.Options(
                "mark" => "none", 
                "color" => get_alg_color(algo.name),
            ),
            Coordinates(coords))
        )
        push!(ps, LegendEntry(algo.name))

        ## Add identification point    
        add_identif_point!(ps, pb, stats, inds, vals, color=get_alg_color(algo.name))

        algo_id += 1
    end

    fig = @pgf Axis(
        {
            height = "12cm",
            width = "12cm",
            xlabel = "proxgrad calls",
            ylabel = L"\|x_k-x_{k-1}\|_2",
            ymode  = "log",
            # xmode  = "log",
            title = "Problem $(pb.name) -- Step length",
            legend_pos = "outer north east",
        },
        ps...
    )
    pgfsave(joinpath(FIGS_FOLDER, "$(pb.name)_step_length.pdf"), fig)
    pgfsave(joinpath(FIGS_FOLDER, "$(pb.name)_step_length.tikz"), fig)
end



function add_contour!(ps, f, x, y)
    push!(ps, PlotInc(
            PGFPlotsX.Options(
                "forget plot" => nothing,
                "no marks" => nothing,
            ),
            Table(contours(x, y, f.(x, y'), 10)),
    ))
    return
end

function add_traj!(ps, coords, algoname; params=Dict())
    if length(coords)>0
        push!(ps, PlotInc(
            PGFPlotsX.Options(
                "smooth" => nothing,
                "thin" => nothing,
                params...
            ),
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

function get_alg_color(algoname)
    res = nothing

    if algoname == "ISTA"
        # res = "red"
        res = distinguishable_colors(9, transform=deuteranopic)[3]
    elseif algoname == "FISTA"
        # res = "blue"
        res = distinguishable_colors(9, transform=deuteranopic)[2+2]
    elseif algoname == "T1"
        # res = "yellow"
        res = distinguishable_colors(9, transform=deuteranopic)[6]
    elseif algoname == "T2"
        res = "black"
        # res = distinguishable_colors(9, transform=deuteranopic)[9]
    elseif algoname == "ISTA - bt"
        res = distinguishable_colors(9, transform=deuteranopic)[1]
    elseif algoname == "FISTA - bt"
        res = distinguishable_colors(9, transform=deuteranopic)[5]
    elseif algoname == "T1 - bt"
        res = distinguishable_colors(9, transform=deuteranopic)[7]
    elseif algoname == "T2 - bt"
        res = distinguishable_colors(9, transform=deuteranopic)[8]
    end

    return res
end

function get_traj_params(algoname)
    res = Dict{Any, Any}()
    if !isnothing(get_alg_color(algoname))
        (res["color"] = get_alg_color(algoname))
        res["mark options"] = "{fill=white}"
    end

    if algoname == "ISTA"
        res["mark"] = "*"
        res["mark size"] = "1pt"
    elseif algoname == "FISTA"
        res["mark"] = "triangle*"
        res["mark size"] = "1pt"
    else
        res["mark size"] = "1.5pt"
        res["fill opacity"] = "0"
    end
    return res
end

function get_legend(algoname)
    res = algoname

    if algoname == "ISTA"
        res = "Proximal Gradient"
    elseif algoname == "FISTA"
        res = "Accel. Proximal Gradient"
    elseif algoname == "T1"
        res = "Prov. Alg -- \$\\mathsf{T}^1\$"
    elseif algoname == "T2"
        res = "Prov. Alg -- \$\\mathsf{T}^2\$"
    end

    return res
end