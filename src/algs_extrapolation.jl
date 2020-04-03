###############################################################################
### identification tests
@inline function identification_test(pb::AbstractProblem, y, y_old, it, mem)
    id = false
    if get(mem, :id_testkind, :default) == :strongtest
        id = id_strongtest(pb::AbstractProblem, y, y_old, it, mem)
    else
        id = id_simpletest(pb::AbstractProblem, y, y_old, it, mem)
    end

    return id
end

## Simple test: checks if new y coordinates are null.
@inline function id_simpletest(pb::AbstractProblem{T}, y, y_old, it, mem) where T<:AbstractRegularizer
    supp_y = get_support(T, y)
    supp_y_old = get_support(T, y_old)
    added_coord = setdiff(supp_y, supp_y_old)

    return length(added_coord) > 0
end

## Strong test: checks if new u coordinates are within interior of (γ∂g + I)(y) for g non diff. at y
@inline function id_strongtest(pb::AbstractProblem{T}, y, y_old, it, mem) where T<:AbstractRegularizer
    reltol = get(mem, :id_reltol, 1e-3) * get(mem, :id_reltol_decay, 1)^(it-1)
    barrier = mem[:α] * pb.λ
    !haskey(mem, :cur_support) && (mem[:cur_support] = SortedSet{Int}())

    ## Compute coordinates strictly within (γ∂g + I)(y)
    strong_support = get_support(T, mem[:u], tol=barrier * (1-reltol))
    support = get_support(T, mem[:u], tol=barrier)

    exclcoords = setdiff(support, strong_support)
    # length(exclcoords)>0 && printstyled("$it \t Excluded coords: $(collect(exclcoords))\n", color=:yellow)

    added_coord = setdiff(strong_support, mem[:cur_support])

    # save new support
    mem[:cur_support] = strong_support

    return length(added_coord) > 0
end

###############################################################################
### Extrapolation functions

## No acceleration
function extra_ISTA(pb::AbstractProblem, y, y_old, it, mem)
    return y
end

## Always acceleration
function extra_FISTA(pb::AbstractProblem, y, y_old, it, mem)
    return y + (mem[:t_old]-1)/mem[:t] * (y-y_old)
end

## Conditional acceleration
function extra_CondInertia(pb::AbstractProblem{T}, y, y_old, it, mem) where T<:AbstractRegularizer
    x_next = y

    if !identification_test(pb, y, y_old, it, mem)
        x_next += (mem[:t_old]-1)/mem[:t] * (y-y_old)
    else
        println("No acceleration this step.")
    end

    return x_next
end

## Predictive conditional acceleration
# Checks wether acceleration degrades future structure wrt to non accelerated structure.
function extra_CondPredInertia(pb::AbstractProblem{T}, y, y_old, it, mem) where T<:AbstractRegularizer
    x_next_pg = y
    x_next_accel = y + (mem[:t_old]-1)/mem[:t] * (y-y_old)

    α = mem[:α]
    T_α_gradprox(pb, x) = prox_αg(pb, x - α * ∇f(pb, x), α)


    pt_pg = T_α_gradprox(pb, x_next_pg)
    pt_accel = T_α_gradprox(pb, x_next_accel)

    support_proxgrad = get_support(T, pt_pg, tol=1e-11)
    support_accel    = get_support(T, pt_accel, tol=1e-11)

    # support_proxgrad !== support_accel && (@show collect(support_proxgrad), collect(support_accel))

    nextpoint = x_next_accel
    if length(setdiff(support_proxgrad, support_accel)) > 0
        println("No acceleration this step.")
        nextpoint = x_next_pg

        # if F(pb, pt_pg) > F(pb, pt_accel)
        #     @printf( "%16e  %16e     %16e\n", F(pb, pt_pg), F(pb, pt_accel), F(pb, pt_pg) - F(pb, pt_accel))
        # end
    end

    return nextpoint
end

## Acceleration if keeps functional decrease
function extra_MFISTA(pb::AbstractProblem, y, y_old, it, mem)
    if it == 1
        mem[:z_old] = y
    end
    z_old = copy(mem[:z_old])

    z = zeros(size(y))
    if argmin([F(pb, y), F(pb, z_old)]) == 1
        z = y
    else
        z = z_old
    end

    mem[:z_old] = copy(z)
    return z + mem[:t_old]/mem[:t] * (y - z) + (mem[:t_old]-1)/mem[:t] * (z-z_old)
end

## Acceleration every other step
# Gives functional decrease every two steps.
function extra_AltIntertia(pb::AbstractProblem, y, y_old, it, mem)
    x_next = y
    if mod(it, 2) == 1
        x_next = y + (mem[:t_old]-1)/mem[:t] * (y-y_old)
    end
    return x_next
end
