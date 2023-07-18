### two-level algorithm functions for dynamic control problem ###
using LinearAlgebra
using SparseArrays
using JuMP
using Ipopt
using Gurobi
using SCS


### update x block in ADMM algorithm
function x_update(x_k, h̄_k, z_k, y_k, λ_n, data, β, ρ, t, scc_time; ρ_scc=50, umin=0.2, δmax=10, resto=false)

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
    ψ_ex(q, A, s; ρ_scc=ρ_scc, umin=umin) = 1/(1+exp(-ρ_scc*((s*q*A) - umin)))

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
    # set_optimizer_attribute(model, "fast_step_computation", "yes")
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
    @NLconstraint(model, [i=1:np], ψ⁺[i] ==  (1+exp(-ρ_scc*((q[i]/1000*A[i]) - umin)))^-1)
    @NLconstraint(model, [i=1:np], ψ⁻[i] ==  (1+exp(-ρ_scc*(-(q[i]/1000*A[i]) - umin)))^-1)

    # objective function
    if obj_type == "azp"
            @objective(model, Min, 
                sum(
                    azp_weights[i]*(h[i] - elev[i])
                    + λ_n[i] * z_k[i]
                    + (β / 2) * (z_k[i])^2 
                    +  y_k[i] * (h[i] - h̄_k[i] + z_k[i]) 
                    + (ρ / 2) * (h[i] - h̄_k[i] + z_k[i])^2
                for i ∈ collect(1:nn))
            )

    elseif obj_type == "scc"
            @objective(model, Min,
                sum(-scc_weights[j]*(ψ⁺[j] + ψ⁻[j]) for j ∈ collect(1:np))
                + sum(
                    λ_n[i] * z_k[i]
                    + (β / 2) * (z_k[i])^2 
                    +  y_k[i] * (h[i] - h̄_k[i] + z_k[i]) 
                    + (ρ / 2) * (h[i] - h̄_k[i] + z_k[i])^2
                for i ∈ collect(1:nn))
            )
    end

    # unload and set starting values
    q_init = x_k[1:np]
    h_init = x_k[np+1:np+nn]
    η_init = x_k[np+nn+1:np+nn+np]
    α_init = x_k[np+nn+np+1:end]
    ψ⁺_init = zeros(np)
    ψ⁻_init = zeros(np)
    for j ∈ 1:np
        ψ⁺_init[j] =  (1+exp(-ρ_scc*((q_init[j]/1000*A[j]) - umin)))^-1
        ψ⁻_init[j] = (1+exp(-ρ_scc*(-(q_init[j]/1000*A[j]) - umin)))^-1
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



### update h̄ block in ADMM algorithm
function h̄_update(x_k, h̄_k, z_k, y_k, λ_n, data, β, ρ, pv_type; δmax=10)

    # unload problem data
    np = data["np"]
    nn = data["nn"]
    nt = data["nt"]
    h_k = x_k[np+1:np+nn, :]
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
        set_optimizer_attribute(model, "OutputFlag", 0)
        set_silent(model)
        @variable(model, h̄[i=1:nn, k=1:nt])
        @constraint(model, [i=1:nn, k=1:nt; k!=nt], h̄[i, k+1] .- h̄[i, k] .≤ δmax)
        @constraint(model, [i=1:nn, k=1:nt; k!=nt], -δmax .≤ h̄[i, k+1] .- h̄[i, k])

        @objective(model, Min, 
            sum(
                sum(
                    λ_n[i, k] * z_k[i, k]
                    + (β / 2) * (z_k[i, k])^2  
                    +  y_k[i, k] * (h_k[i, k] - h̄[i, k] + z_k[i, k]) 
                    + (ρ / 2) * (h_k[i, k] - h̄[i, k] + z_k[i, k])^2
                for i ∈ collect(1:nn))
            for k ∈ collect(1:nt))
            )

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
        @variable(model, Hmin[i, k] ≤ h̄[i=1:nn, k=1:nt] ≤ Hmax[i, k])

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
        @constraint(model, [i=1:nn], vec(h̄[i, :])'*A*vec(h̄[i, :]) .≤ δmax^2)

        @objective(model, Min, 
            sum(
                sum(
                    λ_n[i, k] * z_k[i, k]
                    + (β / 2) * (z_k[i, k])^2  
                    +  y_k[i, k] * (h_k[i, k] - h̄[i, k] + z_k[i, k]) 
                    + (ρ / 2) * (h_k[i, k] - h̄[i, k] + z_k[i, k])^2
                for i ∈ collect(1:nn))
            for k ∈ collect(1:nt))
            )

        set_start_value.(h̄, h̄_k)
 
    elseif pv_type == "range"
        # model = Model(Ipopt.Optimizer)
        # set_optimizer_attribute(model, "warm_start_init_point", "yes")
        # set_optimizer_attribute(model, "linear_solver", "ma57")
        # set_optimizer_attribute(model, "mu_strategy", "adaptive")
        # set_optimizer_attribute(model, "mu_oracle", "quality-function")
        # set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
        # set_optimizer_attribute(model, "print_level", 0)
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)
        set_silent(model)
        @variable(model, h̄[i=1:nn, k=1:nt])
        @variable(model, l[i=1:nn])
        @variable(model, u[i=1:nn])
        @constraint(model, [i=1:nn, k=1:nt], h̄[i, k] ≤ u[i])
        @constraint(model, [i=1:nn, k=1:nt], l[i] ≤ h̄[i, k])
        @constraint(model, [i=1:nn], u[i] - l[i] .≤ δmax)

        @objective(model, Min, 
            sum(
                sum(
                    λ_n[i, k] * z_k[i, k]
                    + (β / 2) * (z_k[i, k])^2  
                    +  y_k[i, k] * (h_k[i, k] - h̄[i, k] + z_k[i, k]) 
                    + (ρ / 2) * (h_k[i, k] - h̄[i, k] + z_k[i, k])^2
                for i ∈ collect(1:nn))
            for k ∈ collect(1:nt))
            )

    elseif pv_type == "none"

        @objective(model, Min, 
            sum(
                sum(
                    λ_n[i, k] * z_k[i, k]
                    + (β / 2) * (z_k[i, k])^2  
                    +  y_k[i, k] * (h_k[i, k] - h̄[i, k] + z_k[i, k]) 
                    + (ρ / 2) * (h_k[i, k] - h̄[i, k] + z_k[i, k])^2
                for i ∈ collect(1:nn))
            for k ∈ collect(1:nt))
            )

    end

    # run optimization model
    optimize!(model)

    return value.(h̄)


end



### update z block in ADMM algorithm (nb: this block is unconstrained)
function z_update(x_k, h̄_k, z_k, y_k, λ_n, data, β, ρ)

    # unload problem data
    np = data["np"]
    nn = data["nn"]
    nt = data["nt"]
    h_k = x_k[np+1:np+nn, :]

    # build model
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_silent(model)
    @variable(model, z[i=1:nn, k=1:nt])
    @objective(model, Min, 
    sum(
        sum(
            λ_n[i, k] * z[i, k]
            + (β / 2) * (z[i, k])^2  
            +  y_k[i, k] * (h_k[i, k] - h̄_k[i, k] + z[i, k]) 
            + (ρ / 2) * (h_k[i, k] - h̄_k[i, k] + z[i, k])^2
        for i ∈ collect(1:nn))
    for k ∈ collect(1:nt))
    )

    # run optimization model
    optimize!(model)

    return value.(z)

end


### update dual variables λ in ALM outer level
function λ_update(λ_n, z_k, β, λ_bound)

    # update λ_n
   temp =  λ_n .+ β .* z_k
   λ_n = min.(max.(temp, -λ_bound), λ_bound)

   return λ_n

end