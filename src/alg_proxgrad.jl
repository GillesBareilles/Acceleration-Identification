###############################################################################
### Main proximal gradient function

"""
    x, history = solve_proxgrad(pb, x0, extrapolation; tol, itmax, x1, logstep, kwargs...)

    Run the proximal gradient algorithm solving `pb`, starting from `x0`, using extrapolation function `extrapolation` and return
    optimal poit `x` and indicators history `history`.

    kwargs holds arguments that will be passed to `extrapolation`.
"""
function solve_proxgrad(pb::AbstractProblem{T},
                        x0,
                        extrapolation::Function;
                        tol = 1e-9,
                        itmax = 5000,
                        x1 = nothing,
                        printstep = nothing,
                        logstep = 1,
                        linesearch = false,
                        αuser = nothing,
                        saveiter = true,
                        proxgrad_per_it = 1,
                        kwargs...) where {T}
    n = pb.n
    L = get_gradlips(pb)
    μ = get_μ_cvx(pb)
    @show L, μ
    α = 1/L
    !isnothing(αuser) && (α = αuser)

    @assert length(x0) == n
    x = copy(x0)
    x_old = zeros(n).+1e9

    y = copy(x)
    y_old = copy(y)
    memory = Dict{Symbol, Any}(
        :t => 1.,
        :t_old => 1.,
        :μ => μ,
        :L => L,
        kwargs...
    )

    it=1

    Lk = linesearch ? 1e-10 : L

    ## Secondary stuff: history and kpis
    isnothing(printstep) && (printstep = floor(itmax / 30))

    history = Dict{Symbol, Any}()
    history[:iter_f] = SortedDict{Int, Float64}(0 => F(pb, x))
    history[:iter_x] = SortedDict{Int, Any}(0 => x)
    history[:iter_y] = SortedDict{Int, Any}(0 => y)
    history[:iter_step_length] = SortedDict{Int, Any}()
    history[:iter_support_x] = SortedDict{Int, Any}()
    history[:iter_support_y] = SortedDict{Int, Any}()
    cur_support = SortedSet{Int64}()

    # Number of identifications, desidentifications
    history[:num_id] = 0
    history[:num_desid] = 0
    history[:desidtime_to_num] = SortedDict{Int, Int}()
    history[:coord_to_idit] = Dict{Int, Int}()


    println("==== Proximal gradient solve")
    println("Problem type:          ", typeof(pb))
    println("Lipschitz constant:    ", L)
    !linesearch && println("Step size:             ", α)
    linesearch && println("Step size:             ", "backtracking")


    print("\nit       obj.                       step               struct x          struct y")
    while norm(x-x_old)>tol && it<itmax
        printtest = length(setdiff(get_support(T, y_old), get_support(T, y))) > 0
        if mod(it, printstep) == 0 || printtest
            logstring = @sprintf "\n%6d   %.19e  %.5e" it F(pb, x) norm(x-x_old)
            logstring2 = @sprintf "%16s  %16s" sprint(show, collect(get_support(T, x))) sprint(show, collect(get_support(T, y)))

            col = printtest ? :red : :green
            printstyled(logstring*logstring2, color=col)
        end

        x_old = copy(x)
        y_old = copy(y)
        memory[:t_old] = memory[:t]
        it == 1 && !isnothing(x1) && (y_old = copy(x1))

        ## Foward Backward
        if linesearch
            Lk = fwbw_backtracking(pb, x, ∇f(pb, x), Lk)
            α = 1/Lk
        end

        u = x - α * ∇f(pb, x)
        y = prox_αg(pb, u, α)

        ## Extrapolation step
        p = 1/20
        q = (p^2 + (2-p)^2)/2
        r = 4.0
        memory[:t] = (p + sqrt(q+r*memory[:t]^2))/2
        memory[:u] = u
        memory[:α] = α

        x = extrapolation(pb, y, y_old, it, memory)

        cur_supp_x = get_support(T, x)
        cur_supp_y = get_support(T, y)

        ## history
        if mod(it-1, logstep) == 0
            ## relevant iterate
            relevant_iterate = x
            if haskey(memory, :evaluate_funcval)
                relevant_iterate = memory[:z_old]
                @printf "%16s" collect(get_support(T, relevant_iterate))
            end
            history[:iter_f][proxgrad_per_it*it] = F(pb, relevant_iterate)
            saveiter && (history[:iter_x][proxgrad_per_it*it] = relevant_iterate)
            saveiter && (history[:iter_y][proxgrad_per_it*it] = y)
            history[:iter_step_length][proxgrad_per_it*it] = norm(x-x_old)
            history[:iter_support_x][proxgrad_per_it*it] = get_support(T, relevant_iterate)
            history[:iter_support_y][proxgrad_per_it*it] = get_support(T, y)
        end

        if cur_supp_x != last(history[:iter_support_x]).second
            history[:iter_support_x][proxgrad_per_it*it] = cur_supp_x
        end

        if cur_supp_y != last(history[:iter_support_y]).second
            history[:iter_support_y][proxgrad_per_it*it] = cur_supp_y
        end

        ## Support
        old_support = cur_support
        cur_support = get_support(T, x)
        newelems = setdiff(cur_support, old_support)
        # !isempty(newelems) && printstyled("$it \tSupport length: $(length(cur_support))\t\tNew elements    : $(collect(newelems))\n", color=:green)

        delelems = setdiff(old_support, cur_support)
        # !isempty(delelems) && printstyled("$it \tSupport length: $(length(cur_support))\t\tRemoved elements: $(collect(delelems))\n", color=:red)

        history[:num_id] += length(newelems)
        history[:num_desid] += length(delelems)

        ## Count number of ins and outs
        for elem in newelems
            @assert !haskey(history[:coord_to_idit], elem)
            history[:coord_to_idit][elem] = it
        end
        for elem in delelems
            desidtime = it-history[:coord_to_idit][elem]

            history[:desidtime_to_num][desidtime] = get(history[:desidtime_to_num], desidtime, 0) + 1
            delete!(history[:coord_to_idit], elem)
        end

        it += 1
    end
    @printf "\n%6d   %.19e  %.5e\n" it F(pb, x) norm(x-x_old)

    support = get_support(T, x)

    println("")
    println("Final solution nzeros: ", length(support), " \t(thresh: 1e-13)")
    print("Null coordinates: "); print(collect(support))
    println()
    println("# identifications      : ", history[:num_id])
    println("#     - faulty id      : ", history[:num_desid])
    println()
    @assert length(history[:coord_to_idit]) == length(support)
    println("Time of false identification to number of occurences:")
    display(collect(history[:desidtime_to_num]))
    # println("Time of identification if final coordinates:")
    # display(collect(history[:coord_to_idit]))

    return x, history
end

function fwbw_backtracking(pb, x, gradf, L)
    Lk = L
    η = 2

    # println("Backtracking -- Lk = $Lk")
    u = x - (1/Lk) * gradf
    y_ls = prox_αg(pb, u, 1/Lk)

    # @assert f(pb, y_ls) ≤ f(pb, x) + dot(gradf, y_ls-x) + Lk/2 * norm(y_ls - x)^2

    k = 0
    while (f(pb, y_ls) > f(pb, x) + dot(gradf, y_ls-x) + Lk/2 * norm(y_ls - x)^2) && (k < 50)
        Lk *= η

        u = x - (1/Lk) * gradf
        y_ls = prox_αg(pb, u, 1/Lk)

        # @show -f(pb, y_ls) + f(pb, x) + dot(gradf, y_ls-x) + Lk/2 * norm(y_ls - x)^2, Lk

        k += 1
    end

    # printstyled("its, final Lk = $k, $Lk\n", color=:red)

    return Lk
end
