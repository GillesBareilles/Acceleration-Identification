export get_iterates_algoparams

function get_alg_color(algoname)
    res = nothing

    if algoname == "ISTA"
        res = "rgb,1:red,0.0;green,0.3843;blue,0.9922"
    elseif algoname == "FISTA"
        res = "rgb,1:red,0.7451;green,0.0;blue,0.0"
    elseif algoname == "T1"
        res = "rgb,1:red,0.0;green,0.502;blue,0.3804"
    elseif algoname == "T2"
        res = "black"
    elseif algoname == "MFISTA"
        res = "rgb,255:red,231;green,84;blue,121"
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

function get_iterates_algoparams(algoname)
    res = Dict{Any, Any}()

    FigsIteratesAlgoCurveswidth = "very thick"

    res[FigsIteratesAlgoCurveswidth] = nothing

    if algoname == "ISTA"
        res = Dict(
            "smooth" => nothing,
            FigsIteratesAlgoCurveswidth => nothing,
            "mark size" => "1pt",
            "color" => get_alg_color(algoname),
            "mark" => "*",
            "mark options" => ""
        )
    elseif algoname == "FISTA"
        res = Dict(
            "smooth" => nothing,
            FigsIteratesAlgoCurveswidth => nothing,
            "mark size" => "1pt",
            "color" => get_alg_color(algoname),
            "mark" => "triangle*",
            "mark options" => ""
        )
    elseif algoname == "T1"
        res = Dict(
            "smooth" => nothing,
            FigsIteratesAlgoCurveswidth => nothing,
            # "fill opacity" => "0",
            "mark size" => "1.5pt",
            "color" => get_alg_color(algoname),
            "mark" => "*",
            "mark options" => ""
        )
    elseif algoname == "T2"
        res = Dict(
            "smooth" => nothing,
            FigsIteratesAlgoCurveswidth => nothing,
            # "fill opacity" => "0",
            "mark size" => "1.5pt",
            "color" => get_alg_color(algoname),
            "mark" => "star",
            # "mark options" => ""
        )
    elseif algoname == "MFISTA"
        res = Dict(
            "smooth" => nothing,
            FigsIteratesAlgoCurveswidth => nothing,
            "mark size" => "1pt",
            "color" => get_alg_color(algoname),
            "mark" => "pentagon*",
            "mark options" => ""
        )
    end


    if !haskey(res, "color")
        (res["color"] = get_alg_color(algoname))
        res["mark options"] = "{fill=white}"
    end

    return res
end


function get_suboptimality_algoparams(algoname)
    res = Dict{Any, Any}(
        "mark" => "none",
        "color" => get_alg_color(algoname),
    )

    if algoname == "ISTA"
        merge!(res, Dict(
            "dotted" => nothing
        ))
    elseif algoname == "FISTA"
        merge!(res, Dict(
            "densely dotted" => nothing
        ))
    elseif algoname == "T1"
        merge!(res, Dict(
            "dashed" => nothing
        ))
    elseif algoname == "T2"
        # merge!(res, Dict(
        # ))
    elseif algoname == "MFISTA"
        # merge!(res, Dict(
        #     "dashed" => nothing
        # ))
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
    elseif algoname == "MFISTA"
        res = "Monotone APG"
    end

    return res
end