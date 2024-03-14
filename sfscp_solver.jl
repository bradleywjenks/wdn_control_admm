### load dependencies ###
begin

    using FileIO
    using Ipopt
    using JLD2
    using JuMP
    using LaTeXStrings
    using LinearAlgebra
    using OpWater
    using Plots
    using Statistics

    include("src/sfscp_functions.jl")

end

########## LOAD DATA ##########

# model information
begin

    script_dir = @__DIR__

    net_name = "bwfl_2022_05_hw"
    # net_name = "modena"

    n_v = 3
    n_f = 4
    δmax = 100
    obj_type = "azp-scc"
    pv_active = true

end

# unload data
begin

    data = load("data/problem_data/"*net_name*"_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2")

    nt = data["nt"]
    np = data["np"]
    nn = data["nn"]
    network = data["network"]
    A13 = spzeros(np)
    A13[data["v_loc"]] .=1
    q_up = data["Qmax"]

end

# set sfscp solver parameters
begin
    max_iter=100
    ϵ_tol=1e-3
end



########## SFSCP SOLVER ##########

# initialize sfscp solver
begin

    # # make starting point 
    # starting_point = "feasible control"
    # q_k, h_k, η_k, α_k, feasible = make_starting_point(data, starting_point, pv_active, δmax)
    # if feasible
    #     obj_k = objective_function(obj_type, data, q_k, h_k)
    #     @info "Starting point is feasible."
    # else
    #     obj_k = Inf
    #     @error "Starting point is not feasible."
    # end

    # load feasibe starting point
    @load "data/WDSA_CCWI_2024/bwfl_2022_05_hw_feasible_starting.jld2" q_k h_k η_k α_k obj_k feasible

    # initialize results matrices
    η_hist = []
    push!(η_hist, η_k)
    α_hist = []
    push!(α_hist, α_k)
    obj_hist = []
    push!(obj_hist, obj_k)
    @info "iter: 0 \t obj_val: $(round(obj_k, digits=2)) \t Ki: -"

    # build optimization model
    convex_model = build_convex_model(data, obj_type, pv_active, δmax, q_k, h_k, η_k, α_k)

end

# run sfscp solver
cpu_time = @elapsed begin

    for kk ∈ collect(1:max_iter)
        
        # step 1: compute line search step
        optimize!(convex_model)
        η_t = value.(convex_model[:η])
        α_t = value.(convex_model[:α])

        # step 2: acceptance of trial point and line search
        q_t, h_t, _, _ = hydraulic_simulation(network, q_up, η_t, A13, α_t)
        obj_t = objective_function(obj_type, data, q_t, h_t)
        feasible = is_feasible(data, pv_active, δmax, q_t, h_t, η_t, α_t)
        γ = 1
        dη = η_t - η_k
        dα = α_t - α_k

        # find acceptable line search step
        while obj_t - obj_k ≥ 0 || !feasible
            γ = 0.5 * γ
            η_t = η_k + γ * dη
            α_t = α_k + γ *dα

            q_t, h_t, _, _ = hydraulic_simulation(network, q_up, η_t, A13, α_t)
            obj_t = objective_function(obj_type, data, q_t, h_t)
            feasible = is_feasible(data, pv_active, δmax, q_t, h_t, η_t, α_t)

            if norm(γ)<1e-4
                break
            end

        end

        @info "γ = $γ"

        if feasible
            obj_old = obj_k
            obj_k = obj_t
            η_k = η_t
            α_k = α_t
            q_k = q_t
            h_k = h_t
            Ki = abs(obj_old - obj_k) / abs(obj_old)

            # update convex model with new linearizations
            # convex_model = update_convex_model(data, convex_model, q_k, h_k, η_k, α_k)
            convex_model = build_convex_model(data, obj_type, pv_active, δmax, q_k, h_k, η_k, α_k)

        else
            Ki = 0
        end

        # save 
        push!(η_hist, η_k)
        push!(α_hist, α_k)
        push!(obj_hist, obj_k)

        @info "iter: $kk \t obj_val: $(round(obj_k, digits=2)) \t Ki: $(round(Ki, digits=4))"

        if Ki ≤ ϵ_tol
            break
        end

    end
    
end


### save data ###
begin
    @save "data/WDSA_CCWI_2024/$(net_name)_pv_$(string(δmax))_scp.jld2" cpu_time obj_k q_k h_k η_k α_k
end

### load data ###
begin
    @load "data/WDSA_CCWI_2024/$(net_name)_pv_$(string(δmax))_scp.jld2" cpu_time obj_k q_k h_k η_k α_k
end
