### functions used in strictly feasible sequential convex programming solver ###
using SparseArrays
using JuMP
using Ipopt
using Gurobi
using FileIO
using JLD2
using OpWater



function is_feasible(data, pv_active, δmax, q, h, η, α)

    # unload problem data
    nn = data["nn"]
    nt = data["nt"]
    q_lo = data["Qmin"]
    q_up = data["Qmax"]
    h_lo = data["Hmin"]
    h_up = data["Hmax"]
    η_lo = data["ηmin"]
    η_up = data["ηmax"]
    α_up = data["αmax"]
    α_lo = zeros(nn, nt)

    # set variable bounds at pcv links  
    v_loc = data["v_loc"]
    v_dir = data["v_dir"]
    for (idx, valve) ∈ enumerate(v_loc)
        if v_dir[idx] == 1
            q_lo[valve, :] .= 0
            η_lo[valve, :] .= 0
        elseif v_dir[idx] == -1
            q_up[v_loc, :] .= 0
            η_up[v_loc, :] .= 0
        end
    end

    # check variable bounds
    q_bounds = all(q_lo .-1e-2 .≤ q .≤ q_up .+ 1e-2)
    h_bounds = all(h_lo .-1e-2 .≤ h .≤ h_up .+ 1e-2)
    η_bounds = all(η_lo .-1e-2 .≤ η .≤ η_up .+ 1e-2)
    α_bounds = all(α_lo .≤ α .≤ α_up)
    η_dir = all(-1e-2 .≤ q.*η)
    pv_constraint = all([maximum(h[i, :]) - minimum(h[i, :]) for i in 1:nn] .≤ δmax)

    # are any bounds violated?
    if pv_active
        all_feas = all((q_bounds, h_bounds, η_bounds, η_dir, α_bounds, pv_constraint))
    else
        all_feas = all((q_bounds, h_bounds, η_bounds, η_dir, α_bounds))
    end

    return all_feas

end



function objective_function(obj_type, data, q, h)

    # unload problem data
    nt = data["nt"]
    np, nn = size(data["A12"])
    scc_time = data["scc_time"]
    A = 1 ./ ((π/4).*data["D"].^2)
    elev = data["elev"]
    ρ = data["ρ"]
    umin = data["umin"]

    if obj_type == "scc"
        scc_weights = repeat(data["scc_weights"], 1, nt)
        azp_weights = zeros(nn, nt)
    elseif obj_type == "azp"
        azp_weights = repeat(data["azp_weights"], 1, nt)
        scc_weights = zeros(np, nt)
    elseif obj_type == "azp-scc"
        scc_weights = zeros(np, nt)
        scc_weights[:, scc_time] .= data["scc_weights"]
        azp_weights = zeros(nn, nt)
        azp_weights[:, setdiff(collect(1:nt), scc_time)] .= data["azp_weights"]
    end

    f_scc = sum(
        sum(
            -scc_weights[j, k]*(
                1/(1+exp(-ρ*((1*q[j, k]/1000*A[j]) - umin))) +
                1/(1+exp(-ρ*((-1*q[j, k]/1000*A[j]) - umin))) 
            ) 
        for j ∈ 1:np) 
    for k ∈ 1:nt)

    f_azp = sum(
        sum(
            azp_weights[i, k]*(
                h[i, k] - elev[i]
            ) 
        for i ∈ 1:nn) 
    for k ∈ 1:nt)

    if obj_type == "scc" || obj_type == "azp"
        f = (f_scc + f_azp) ./ nt
    else
        f = f_scc + f_azp
    end

    return f
    
end



function make_starting_point(data, starting_point, pv_active, δmax)

    #unload data
    network = data["network"]
    v_loc = data["v_loc"]
    v_dir = data["v_dir"]
    q_up = data["Qmax"]
    np = data["np"]
    nt = data["nt"]
    A13 = spzeros(np)
    A13[v_loc] .=1

    if starting_point == "no control"

        q = data["q_init"]
        h = data["h_init"]
        η = zeros(data["np"], data["nt"])
        α = zeros(data["nn"], data["nt"])

    elseif starting_point == "feasible control"

        q, h, η, α = ipopt_solver(data)

    end

    # check feasibility
    feasible = is_feasible(data, pv_active, δmax, q, h, η, α)

    return q, h, η, α, feasible

end




function build_convex_model(data, obj_type, pv_active, δmax, q_k, h_k, η_k, α_k)

    # unload data
    nt = data["nt"]
    elev = data["elev"]
    A10 = data["A10"]
    A12 = data["A12"]
    np, nn = size(A12)
    d = data["d"]
    h0 = data["h0"]
    A = 1 ./ ((π/4).*data["D"].^2)
    scc_time = data["scc_time"]
    r = data["r"]
    nexp = data["nexp"]
    v_loc = data["v_loc"]
    v_dir = data["v_dir"]
    q_lo = data["Qmin"]
    q_up = data["Qmax"]
    h_lo = data["Hmin"]
    h_up = data["Hmax"]
    η_lo = data["ηmin"]
    η_up = data["ηmax"]
    α_up = data["αmax"]
    α_lo = zeros(nn, nt)
    ρ = data["ρ"]
    umin = data["umin"]
    azp_weights = data["azp_weights"]
    scc_weights = data["scc_weights"]

    # find junction and source nodes
    nodes_map = Dict(i=> findall(A12[i, :].!=0) for i in 1:np)
    sources_map = Dict(i=> findall(A10[i, :].!=0) for i in 1:np)

    # set variable bounds at pcv links  
    for (idx, valve) ∈ enumerate(v_loc)
        if v_dir[idx] == 1
            q_lo[valve, :] .= 0
            η_lo[valve, :] .= 0
        elseif v_dir[idx] == -1
            q_up[v_loc, :] .= 0
            η_up[v_loc, :] .= 0
        end
    end

    # define functions
    ϕ(q, r, nexp) = r*q*abs.(q)^(nexp-1)
    ∇ϕ_q(q, r, nexp) = nexp*r*abs.(q)^(nexp-1)
    ψ(q, A, s; ρ=ρ, umin=umin) = 1/(1+exp(-ρ*((s*q./1000*A) - umin)))
    ∇ψ_q(q, A, s; ρ=ρ, umin=umin) = (s*ρ*A)*exp(-ρ*((s*q./1000*A)-umin))*(1/(1+exp(-ρ*((s*q./1000*A)-umin)))^2)

    # build model
    model = Model(Gurobi.Optimizer)
    #set_optimizer_attribute(model,"Method", 2)
    # set_optimizer_attribute(model,"Presolve", 0)
    # set_optimizer_attribute(model,"Crossover", 0)
    #set_optimizer_attribute(model,"NumericFocus", 3)
    # set_optimizer_attribute(model,"NonConvex", 2)
    # set_silent(model)
    
    # define variables
    @variable(model, q_lo[i, k] ≤ q[i=1:np, k=1:nt] ≤ q_up[i, k])
    @variable(model, h_lo[i, k] ≤ h[i=1:nn, k=1:nt] ≤ h_up[i, k])
    @variable(model, η_lo[i, k] ≤ η[i=1:np, k=1:nt] ≤ η_up[i, k])
    @variable(model, α_lo[i, k] ≤ α[i=1:nn, k=1:nt] ≤ α_up[i, k])
    @variable(model, ψ⁺[i=1:np, k=1:nt])
    @variable(model, ψ⁻[i=1:np, k=1:nt])

    # set objective function
    if obj_type=="azp"
        @objective(model, Min, (1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt))

    elseif obj_type =="scc"
        @objective(model, Min, (1/nt)*sum(sum(-scc_weights[i]*(ψ⁺[i, k] + ψ⁻[i, k]) for i ∈ 1:np) for k ∈ 1:nt))

    elseif obj_type == "azp-scc"
        scc_weights = zeros(np, nt)
        scc_weights[:, scc_time] .= data["scc_weights"]
        azp_weights = zeros(nn, nt)
        azp_weights[:, setdiff(collect(1:nt), scc_time)] .= data["azp_weights"]
    
        @objective(model, Min, 
            (sum(
                sum(
                    -scc_weights[i, k]*(
                        ψ⁺[i, k] + ψ⁻[i, k]
                    ) 
                for i ∈ 1:np) 
            for k ∈ 1:nt)
            ) +
            (sum(
                sum(
                    azp_weights[i, k]*(
                        h[i, k] - elev[i]
                    ) 
                for i ∈ 1:nn) 
            for k ∈ 1:nt)
            )
        )

    end

    # linear energy and mass constraint
    a_k = [ϕ(q_k[i, t], r[i], nexp[i]) * (1-nexp[i]) for i in 1:np, t in 1:nt]
    b_k = [∇ϕ_q(q_k[i, t], r[i], nexp[i]) for i in 1:np, t in 1:nt] 
    @constraint(model, [i=1:np, k=1:nt], a_k[i, k] + b_k[i, k]*q[i, k] + sum(A12[i, j]*h[j, k] for j ∈ nodes_map[i]) + sum(A10[i, j]*h0[j, k] for j ∈ sources_map[i]) + η[i, k] == 0)
    @constraint(model, A12'*q - α .== d)

    # # bilinear control valve constraint
    # ϵ_bi = 0 # constraint violation tolerance
    # @constraint(model, linearised_bi[i=1:np, k=1:nt; i in v_loc], η_k[i, k]*q[i, k] + q_k[i, k]*η[i, k] ≥ ϵ_bi + q_k[i, k]*η_k[i, k])

    # compute linearised f_scc objective value terms
    @constraint(model, linearised_ψ⁺[i=1:np, k=1:nt], ψ⁺[i, k] - ∇ψ_q(q_k[i, k]/1000, A[i], 1)*q[i, k] == ψ(q_k[i, k]/1000, A[i], 1) - ∇ψ_q(q_k[i, k]/1000, A[i], 1)*q_k[i, k])
    @constraint(model, linearised_ψ⁻[i=1:np, k=1:nt], ψ⁻[i, k] - ∇ψ_q(q_k[i, k]/1000, A[i], -1)*q[i, k] == ψ(q_k[i, k]/1000, A[i], -1) - ∇ψ_q(q_k[i, k]/1000, A[i], -1)*q_k[i, k])

    if pv_active
        # pressure range constraint
        @variable(model, l[i=1:nn])
        @variable(model, u[i=1:nn])
        @constraint(model, [i=1:nn, k=1:nt], h[i, k] ≤ u[i])
        @constraint(model, [i=1:nn, k=1:nt], l[i] ≤ h[i, k])
        @constraint(model, [i=1:nn], u[i] - l[i] .≤ δmax)
    end

    return model

end



function update_convex_model(data, convex_model, q_k, h_k, η_k, α_k)

    # unload data
    nt = data["nt"]
    A12 = data["A12"]
    np, nn = size(A12)
    A = 1 ./ ((π/4).*data["D"].^2)
    r = data["r"]
    nexp = data["nexp"]
    v_loc = data["v_loc"]
    ρ = data["ρ"]
    umin = data["umin"]

    # define functions
    ϕ(q_k, r, nexp) = r*q_k*abs.(q_k)^(nexp-1)
    ∇ϕ_q(q_k, r, nexp) = nexp*r*abs.(q_k)^(nexp-1)
    ψ(q_k, A, s; ρ=ρ, umin=umin) = 1/(1+exp(-ρ*((s*q_k./1000*A) - umin)))
    ∇ψ_q(q_k, A, s; ρ=ρ, umin=umin) = (s*ρ*A)*exp(-ρ*((s*q_k./1000*A)-umin))*(1/(1+exp(-ρ*((s*q_k./1000*A)-umin)))^2)

    # update model parameters
    # Threads.@threads for k ∈ collect(1:nt)
    for k ∈ collect(1:nt)
        for i ∈ collect(1:np)

            # linearised energy conservation constraint
            a_k = ϕ(q_k[i, k], r[i], nexp[i]) * (1 - nexp[i])
            b_k = ∇ϕ_q(q_k[i, k], r[i], nexp[i]) .* -1
            set_normalized_coefficient(convex_model[:linearised_hyd][i, k], convex_model[:q][i, k], b_k)
            set_normalized_rhs(convex_model[:linearised_hyd][i, k], a_k)

            # if i ∈ v_loc
            #     # linearised bilinear control valve constraint
            #     ϵ_bi = 0 # constraint violation tolerance
            #     set_normalized_coefficient(convex_model[:linearised_bi][i, k], convex_model[:q][i, k], η_k[i, k])
            #     set_normalized_coefficient(convex_model[:linearised_bi][i, k], convex_model[:η][i, k], q_k[i, k])
            #     set_normalized_rhs(convex_model[:linearised_bi][i, k], ϵ_bi + q_k[i, k]*η_k[i, k])
            # end

            # linearised ψ⁺ equality constraint
            set_normalized_coefficient(convex_model[:linearised_ψ⁺][i, k], convex_model[:q][i, k], -∇ψ_q(q_k[i, k]/1000, A[i], 1))
            set_normalized_rhs(convex_model[:linearised_ψ⁺][i, k], ψ(q_k[i, k]/1000, A[i], 1) - ∇ψ_q(q_k[i, k]/1000, A[i], 1)*q_k[i, k])

            # linearised ψ⁻ equality constraint
            set_normalized_coefficient(convex_model[:linearised_ψ⁻][i, k], convex_model[:q][i, k], -∇ψ_q(q_k[i, k]/1000, A[i], -1))
            set_normalized_rhs(convex_model[:linearised_ψ⁻][i, k], ψ(q_k[i, k]/1000, A[i], -1) - ∇ψ_q(q_k[i, k]/1000, A[i], -1)*q_k[i, k])
        end
    end

    return convex_model
    
end



function ipopt_solver(data)

     # unload data
    elev = data["elev"]
    nexp = data["nexp"]
    A10 = data["A10"]
    A12 = data["A12"]
    np, nn = size(A12)
    nt = data["nt"]
    d = data["d"]
    h0 = data["h0"]
    r = data["r"]
    A = 1 ./ ((π/4).*data["D"].^2)
    q_lo = data["Qmin"]
    q_up = data["Qmax"]
    h_lo = data["Hmin"]
    h_up = data["Hmax"]
    η_lo = data["ηmin"]
    η_up = data["ηmax"]
    α_up = data["αmax"]
    α_lo = zeros(nn, nt)
    y_loc = data["y_loc"]
    v_loc = data["v_loc"]
    v_dir = data["v_dir"]

    # set variable bounds at pcv links
    for (idx, valve) ∈ enumerate(v_loc)
        if v_dir[idx] == 1
            q_lo[valve, :] .= 0
            η_lo[valve, :] .= 0
        elseif v_dir[idx] == -1
            q_up[v_loc, :] .= 0
            η_up[v_loc, :] .= 0
        end
    end

    # find junction and source nodes
    nodes_map = Dict(i=> findall(A12[i, :].!=0) for i in 1:np)
    sources_map = Dict(i=> findall(A10[i, :].!=0) for i in 1:np)

    # set optimizaiton solver and its attributes
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 3000)
    set_optimizer_attribute(model, "linear_solver", "ma57")
    # set_optimizer_attribute(model, "ma57_pivtol", 1e-8)
    # set_optimizer_attribute(model, "ma57_pre_alloc", 10.0)
    # set_optimizer_attribute(model, "ma57_automatic_scaling", "yes")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "mu_oracle", "quality-function")
    set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
    # set_optimizer_attribute(model, "constr_viol_tol", 1e-9)
    # set_optimizer_attribute(model, "fast_step_computation", "yes")
    # set_optimizer_attribute(model, "hessian_approximation", "exact")
    # set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    # set_optimizer_attribute(model, "derivative_test", "first-order")
    # set_optimizer_attribute(model, "derivative_test", "second-order")
    set_optimizer_attribute(model, "print_level", 5)

    # define variables
    @variable(model, q_lo[i, k] ≤ q[i=1:np, k=1:nt] ≤ q_up[i, k])
    @variable(model, h_lo[i, k] ≤ h[i=1:nn, k=1:nt] ≤ h_up[i, k])
    @variable(model, η_lo[i, k] ≤ η[i=1:np, k=1:nt] ≤ η_up[i, k])

    # hydraulic conservation constraints
    reg = 1e-08
    @constraint(model, [i=1:np, k=1:nt], r[i]*(q[i, k]+reg)*abs(q[i, k]+reg)^(nexp[i]-1) + sum(A12[i, j]*h[j, k] for j ∈ nodes_map[i]) + sum(A10[i, j]*h0[j, k] for j ∈ sources_map[i]) + η[i, k] == 0)
    @constraint(model, A12'*q .== d)

    # no objective function
    @objective(model, Min, sum(sum((h[i, k+1] - h[i, k])^2 for k ∈ 1:nt-1) for i ∈ 1:nn))
    # @objective(model, Min, sum(maximum(h[i, :]) - minimum(h[i, :]) for i ∈ 1:nn))

    # maximum runtime
    set_time_limit_sec(model, 6*60*60)

    # run optimization model
    optimize!(model)
    solution_summary(model)
    term_status = termination_status(model)

    accepted_status = [LOCALLY_SOLVED; ALMOST_LOCALLY_SOLVED; OPTIMAL; ALMOST_OPTIMAL]

    if term_status in accepted_status
        q_sol = value.(q)
        h_sol = value.(h)
        η_sol = value.(η)
    else
        q_sol = NaN
        h_sol = NaN
        η_sol = NaN
    end

    return value.(q), value.(h), value.(η), zeros(nn, nt)

end