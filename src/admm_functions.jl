### admm functions for adaptive control problem
using LinearAlgebra
using SparseArrays
using JuMP
using Ipopt
using Gurobi
using MosekTools
using SCS



function primal_update(xk, zk, λk, data, γ, t, scc_time; ρ=50, umin=0.2, δmax=10, resto=false, bi_dir=true, scaled=false)

    # set objective function
    if t ∈ scc_time
        obj_type = "scc"
    else
        obj_type = "azp"
    end
    
    # unload data
    elev = data["elev"]
    nexp = data["nexp"]
    d = data["d"][:, t]
    h0 = data["h0"][:, t]
    A10 = data["A10"]
    A12 = data["A12"]
    np, nn = size(A12)
    nt = data["nt"]
    r = data["r"]
    A = 1 ./ ((π/4).*data["D"].^2)
    azp_weights = data["azp_weights"]
    scc_weights = data["scc_weights"]
    q_lo = data["Qmin"][:, t]
    q_up = data["Qmax"][:, t]
    h_lo = data["Hmin"][:, t]
    h_up = data["Hmax"][:, t]
    η_lo = data["ηmin"][:, t]
    η_up = data["ηmax"][:, t]
    α_up = data["αmax"][:, t]
    α_lo = zeros(nn, 1)
    y_loc = data["y_loc"]
    v_loc = data["v_loc"]

    # define nonlinear SCC objective function
    ψ_ex(q, A, s; ρ=ρ, umin=umin) = 1/(1+exp(-ρ*((s*q*A) - umin)))

    # find junction and source nodes
    nodes_map = Dict(i=> findall(A12[i, :].!=0) for i in 1:np)
    sources_map = Dict(i=> findall(A10[i, :].!=0) for i in 1:np)

    # set optimizaiton solver and its attributes
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 3000)
    set_optimizer_attribute(model, "warm_start_init_point", "yes")
    set_optimizer_attribute(model, "linear_solver", "ma57")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "mu_oracle", "quality-function")
    set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
    # set_optimizer_attribute(model, "tol", 1e-6)
    # set_optimizer_attribute(model, "constr_viol_tol", 1e-9)
    set_optimizer_attribute(model, "fast_step_computation", "yes")
    set_optimizer_attribute(model, "print_level", 0)
    if resto
        set_optimizer_attribute(model, "start_with_resto", "yes")
        set_optimizer_attribute(model, "expect_infeasible_problem", "yes")
    end

    # define variables
    @variable(model, q_lo[i] ≤ q[i=1:np] ≤ q_up[i])
    @variable(model, h_lo[i] ≤ h[i=1:nn] ≤ h_up[i])
    @variable(model, η_lo[i] ≤ η[i=1:np] ≤ η_up[i])
    @variable(model, α_lo[i] ≤ α[i=1:nn] ≤ α_up[i])
    @variable(model, ψ⁺[i=1:np])
    @variable(model, ψ⁻[i=1:np])

    # hydraulic conservation constraints
    reg = 1e-08
    @NLconstraint(model, [i=1:np], r[i]*(q[i]+reg)*abs(q[i]+reg)^(nexp[i]-1) + sum(A12[i, j]*h[j] for j ∈ nodes_map[i]) + sum(A10[i, j]*h0[j] for j ∈ sources_map[i]) + η[i] == 0)
    @constraint(model, A12'*q - α .== d)

    # bilinear constraint for dbv control direction
    ϵ_bi = 0
    @NLconstraint(model, [i=1:np; i in v_loc], ϵ_bi ≤ η[i]*q[i])

    # auxiliary variables for scc sigmoid functions
    @NLconstraint(model, [i=1:np], ψ⁺[i] ==  (1+exp(-ρ*((q[i]/1000*A[i]) - umin)))^-1)
    @NLconstraint(model, [i=1:np], ψ⁻[i] ==  (1+exp(-ρ*(-(q[i]/1000*A[i]) - umin)))^-1)

    # objective function
    if obj_type == "azp"
        if scaled
            if all(λk .== 0)
                uk = zeros(size(λk))
            else
                uk = (1/γ).*λk # scaled dual variable -- see Section 3.1.1 of Boyd et al. (2010)
            end
            @objective(model, Min, 
                sum(azp_weights[i]*(h[i] - elev[i]) + (γ/2)*(h[i] - zk[i] + uk[i])^2 for i ∈ collect(1:nn))
                )
        else
            @objective(model, Min, 
            sum(azp_weights[i]*(h[i] - elev[i]) + λk[i]*(h[i] - zk[i]) + (γ/2)*(h[i] - zk[i])^2 for i ∈ collect(1:nn))
            )
        end

    elseif obj_type == "scc"
        if scaled
            if all(λk .== 0)
                uk = zeros(size(λk))
            else
                uk = (1/γ).*λk # scaled dual variable -- see Section 3.1.1 of Boyd et al. (2010)
            end
            @objective(model, Min,
                sum(-scc_weights[j]*(ψ⁺[j] + ψ⁻[j]) for j ∈ collect(1:np)) + sum((γ/2)*(h[i] - zk[i] + uk[i])^2 for i ∈ collect(1:nn))
                )
        else
            @objective(model, Min,
            sum(-scc_weights[j]*(ψ⁺[j] + ψ⁻[j]) for j ∈ collect(1:np)) + sum(λk[i]*(h[i] - zk[i]) + (γ/2)*(h[i] - zk[i])^2 for i ∈ collect(1:nn))
            )
        end
    end

    # unload and set starting values
    q_init = xk[1:np]
    h_init = xk[np+1:np+nn]
    η_init = xk[np+nn+1:np+nn+np]
    α_init = xk[np+nn+np+1:end]
    ψ⁺_init = zeros(np)
    ψ⁻_init = zeros(np)
    for j ∈ 1:np
        ψ⁺_init[j] =  (1+exp(-ρ*((q_init[j]/1000*A[j]) - umin)))^-1
        ψ⁻_init[j] = (1+exp(-ρ*(-(q_init[j]/1000*A[j]) - umin)))^-1
    end
    set_start_value.(ψ⁺, ψ⁺_init)
    set_start_value.(ψ⁻, ψ⁻_init)
    set_start_value.(q, q_init)
    set_start_value.(h, h_init)
    set_start_value.(η, η_init)
    set_start_value.(α, α_init)

    # run optimization model
    optimize!(model)
    solution_summary(model)
    term_status = termination_status(model)

    accepted_status = [LOCALLY_SOLVED; ALMOST_LOCALLY_SOLVED; OPTIMAL; ALMOST_OPTIMAL]

    if term_status in accepted_status
        x_sol = vcat(value.(q), value.(h), value.(η), value.(α))
        obj_sol = objective_value(model)
        status = 0
    else
        x_sol = repeat([NaN], 2*np+2*nn, 1)
        obj_sol = Inf
        status = 1
    end

    return x_sol, obj_sol, status

end



function  auxiliary_update(xk, zk, λk, data, γ, pv_type; δmax=10, scaled=false)

    # unload problem data
    np = data["np"]
    nn = data["nn"]
    nt = data["nt"]
    hk = xk[np+1:np+nn, :]
    Hmin = data["Hmin"]
    Hmax = data["Hmax"]

    # create optimizaiton problem
    if pv_type == "variation"
        # model = Model(Ipopt.Optimizer)
        # set_optimizer_attribute(model, "warm_start_init_point", "yes")
        # set_optimizer_attribute(model, "linear_solver", "ma57")
        # set_optimizer_attribute(model, "mu_strategy", "adaptive")
        # set_optimizer_attribute(model, "mu_oracle", "quality-function")
        # set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
        # set_optimizer_attribute(model, "print_level", 0)
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        @variable(model, z[i=1:nn, k=1:nt])
        @constraint(model, [i=1:nn, k=1:nt; k!=nt], z[i, k+1] .- z[i, k] .≤ δmax)
        @constraint(model, [i=1:nn, k=1:nt; k!=nt], -δmax .≤ z[i, k+1] .- z[i, k])
        if scaled
            if all(λk .== 0)
                uk = zeros(size(λk))
            else
                uk = (1/γ).*λk # scaled dual variable -- see Section 3.1.1 of Boyd et al. (2010)
            end
            @objective(model, Min, sum(sum((γ/2)*(hk[i, k] - z[i, k] + uk[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt)))
        else
            @objective(model, Min, sum(sum(λk[i, k]*(hk[i, k] - z[i, k]) + (γ/2)*(hk[i, k] - z[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt)))
        end

    elseif pv_type == "variability"
        reg = 1e-12
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "warm_start_init_point", "yes")
        set_optimizer_attribute(model, "linear_solver", "ma57")
        set_optimizer_attribute(model, "mu_strategy", "adaptive")
        set_optimizer_attribute(model, "mu_oracle", "quality-function")
        set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
        set_optimizer_attribute(model, "print_level", 0)

        # model = Model(Gurobi.Optimizer)
        # model = Model(Mosek.Optimizer)
        # model = Model(SCS.Optimizer)
        # set_silent(model)
        @variable(model, Hmin[i, k] ≤ z[i=1:nn, k=1:nt] ≤ Hmax[i, k])

        A = spzeros(nt, nt)
        for i ∈ 1:nt
            if i == 1
                A[i, i] = 1 + reg
                A[i, i+1] = -1
            elseif i == nt
                A[i, i] = 1 + reg
                A[i, i-1] = -1
            else
                A[i, i] = 2 + reg
                A[i, i-1] = -1
                A[i, i+1] = -1
            end
        end
        @constraint(model, [i=1:nn], vec(z[i, :])'*A*vec(z[i, :]) .≤ δmax^2)

        # for i ∈ 2:nt
        #     A[i, i] = 1
        #     A[i, i-1] = -1
        # end
        # @constraint(model, [i=1:nn], [δmax; A*z[i, :]] in SecondOrderCone())

        if scaled
            if all(λk .== 0)
                uk = zeros(size(λk))
            else
                uk = (1/γ).*λk # scaled dual variable -- see Section 3.1.1 of Boyd et al. (2010)
            end
            @objective(model, Min, sum(sum((γ/2)*(hk[i, k] - z[i, k] + uk[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt)))
        else
            @objective(model, Min, sum(sum(λk[i, k]*(hk[i, k] - z[i, k]) + (γ/2)*(hk[i, k] - z[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt)))
        end

        set_start_value.(z, zk)
 
    elseif pv_type == "range"
        # model = Model(Ipopt.Optimizer)
        # set_optimizer_attribute(model, "warm_start_init_point", "yes")
        # set_optimizer_attribute(model, "linear_solver", "ma57")
        # set_optimizer_attribute(model, "mu_strategy", "adaptive")
        # set_optimizer_attribute(model, "mu_oracle", "quality-function")
        # set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
        # set_optimizer_attribute(model, "print_level", 0)
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        @variable(model, z[i=1:nn, k=1:nt])
        @variable(model, l[i=1:nn])
        @variable(model, u[i=1:nn])
        @constraint(model, [i=1:nn, k=1:nt], z[i, k] ≤ u[i])
        @constraint(model, [i=1:nn, k=1:nt], l[i] ≤ z[i, k])
        @constraint(model, [i=1:nn], u[i] - l[i] .≤ δmax)
        if scaled
            if all(λk .== 0)
                uk = zeros(size(λk))
            else
                uk = (1/γ).*λk # scaled dual variable -- see Section 3.1.1 of Boyd et al. (2010)
            end
            @objective(model, Min, sum(sum((γ/2)*(hk[i, k] - z[i, k] + uk[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt)))
        else
            @objective(model, Min, sum(sum(λk[i, k]*(hk[i, k] - z[i, k]) + (γ/2)*(hk[i, k] - z[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt)))
        end

    elseif pv_type == "none"
        if scaled
            if all(λk .== 0)
                uk = zeros(size(λk))
            else
                uk = (1/γ).*λk # scaled dual variable -- see Section 3.1.1 of Boyd et al. (2010)
            end
            @objective(model, Min, sum(sum((γ/2)*(hk[i, k] - z[i, k] + uk[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt)))
        else
            @objective(model, Min, sum(sum(λk[i, k]*(hk[i, k] - z[i, k]) + (γ/2)*(hk[i, k] - z[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt)))
        end
    end

    # run optimization model
    optimize!(model)

    return value.(z)

end


function make_object_data(net_name, network, opt_params, v_loc, v_dir, y_loc, scc_time)
    ### make new object to pass into distributed Julia routine.

    # unload network data
    elev = deepcopy(network.elev)
    nexp = deepcopy(network.nexp)
    d = deepcopy(network.d)
    h0 = deepcopy(network.h0)
    A10 = deepcopy(network.A10)
    A12 = deepcopy(network.A12)
    r = deepcopy(network.r)
    D = deepcopy(network.D)
    nt = deepcopy(network.nt)
    np = deepcopy(network.np)
    nn = deepcopy(network.nn)

    # unload problem data
    Qmin = deepcopy(opt_params.Qmin)
    Qmax = deepcopy(opt_params.Qmax)
    Hmin = deepcopy(opt_params.Hmin)
    Hmax = deepcopy(opt_params.Hmax)
    ηmin = deepcopy(opt_params.ηmin)
    ηmax = deepcopy(opt_params.ηmax)
    αmax = deepcopy(opt_params.αmax)
    azp_weights = deepcopy(opt_params.azp_weights)
    scc_weights = deepcopy(opt_params.scc_weights)
    ρ = deepcopy(opt_params.ρ)
    umin = deepcopy(opt_params.umin)

    # hydraulic_simulation from OpWater package
    q_init, h_init, err, iter = hydraulic_simulation(network, opt_params)

    # assign control valve bounds
    if length(v_loc) > 0
        ηmin[setdiff(1:np, v_loc), :] .= 0
        ηmax[setdiff(1:np, v_loc), :] .= 0
    else
        ηmin .= 0
        ηmax .= 0
    end

    # afv actuators
    if length(y_loc) > 0 
        αmax[setdiff(1:nn, y_loc), :] .= 0
        αmax[:, setdiff(1:nt, scc_time)] .= 0
    else
        αmax .= 0
    end

    @save "data/problem_data/"*net_name*"_nv_"*string(length(v_loc))*"_nf_"*string(length(y_loc))*".jld2" elev nexp d h0 A10 A12 r D nt np nn Qmin Qmax Hmin Hmax ηmin ηmax αmax azp_weights scc_weights v_loc y_loc q_init h_init ρ umin scc_time

end