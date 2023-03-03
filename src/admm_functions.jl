### admm functions for adaptive control problem
using FLoops
using LinearAlgebra
using SparseArrays
using StatsBase
using Parameters



function primal_update(xk, zk, λk, network, probdata, γ, t, scc_time, z_loc, y_loc; ρ=50, umin=0.2, δmax=10)

    # set objective function
    if t ∈ scc_time
        obj_type = "scc"
    else
        obj_type = "azp"
    end
    
    # unload network data
    elev = network.elev
    nexp = network.nexp
    d = network.d[:, t]
    h0 = network.h0[:, t]
    A10 = network.A10
    A12 = network.A12
    np, nn = size(A12)
    azp_weights = probdata.azp_weights
    scc_weights = probdata.scc_weights
    r = determine_r(network.L, network.D, network.C, network.nexp, valve_idx=network.valve_idx)
    A = 1 ./ ((π/4).*network.D.^2)

    # unload problem data
    q_lo = copy(probdata.Qmin[:, t])
    q_up = copy(probdata.Qmax[:, t])
    h_lo = copy(probdata.Hmin[:, t])
    h_up = copy(probdata.Hmax[:, t])
    η_lo = copy(probdata.ηmin[:, t])
    η_up = copy(probdata.ηmax[:, t])
    α_up = copy(probdata.αmax[:, t])
    α_lo = zeros(nn, 1)

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
    # set_optimizer_attribute(model, "start_with_resto", "yes")
    # set_optimizer_attribute(model, "expect_infeasible_problem", "yes")
    # set_optimizer_attribute(model, "tol", 1e-6)
    # set_optimizer_attribute(model, "constr_viol_tol", 1e-9)
    set_optimizer_attribute(model, "fast_step_computation", "yes")
    # set_optimizer_attribute(model, "hessian_approximation", "exact")
    # set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    # set_optimizer_attribute(model, "derivative_test", "first-order")
    # set_optimizer_attribute(model, "derivative_test", "second-order")
    set_optimizer_attribute(model, "print_level", 5)
    set_optimizer_attribute(model, "start_with_resto", "yes")
    set_optimizer_attribute(model, "expect_infeasible_problem", "yes")

    # define variables
    @variable(model, q_lo[i] ≤ q[i=1:np] ≤ q_up[i])
    @variable(model, h_lo[i] ≤ h[i=1:nn] ≤ h_up[i])
    @variable(model, η_lo[i] ≤ η[i=1:np] ≤ η_up[i])
    @variable(model, α_lo[i] ≤ α[i=1:nn] ≤ α_up[i])
    @variable(model, ψ⁺[i=1:np])
    @variable(model, ψ⁻[i=1:np])

     # hydraulic conservation constraints
     @NLconstraint(model, [i=1:np], r[i]*q[i]*abs(q[i])^(nexp[i]-1) + sum(A12[i, j]*h[j] for j ∈ nodes_map[i]) + sum(A10[i, j]*h0[j] for j ∈ sources_map[i]) + η[i] == 0)
    @constraint(model, A12'*q - α .== d)

    # bilinear constraint for dbv control direction
    ϵ_bi = -0.01
    @NLconstraint(model, [i=1:np; i in z_loc], ϵ_bi ≤ η[i]*q[i])

    # auxiliary variables for scc sigmoid functions
    @NLconstraint(model, [i=1:np], ψ⁺[i] ==  (1+exp(-ρ*((q[i]/1000*A[i]) - umin)))^-1)
    @NLconstraint(model, [i=1:np], ψ⁻[i] ==  (1+exp(-ρ*(-(q[i]/1000*A[i]) - umin)))^-1)

    # objective function
    if obj_type == "azp"
        @objective(model, Min, 
            sum(azp_weights[i]*(h[i] - elev[i]) + λk[i]*h[i] + (γ/2)*(h[i] - zk[i])^2 for i ∈ collect(1:nn))
            )

    elseif obj_type == "scc"
        @objective(model, Min,
            sum(-scc_weights[j]*(ψ⁺[j] + ψ⁻[j]) for j ∈ collect(1:np)) + sum(λk[i]*h[i] + (γ/2)*(h[i] - zk[i])^2 for i ∈ collect(1:nn))
            )
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


function  auxiliary_update(xk, zk, λk, network, probdata, γ; δmax=10)

    # unload problem data
    z_lo = copy(probdata.Hmin)
    z_up = copy(probdata.Hmax)
    np = network.np
    nn = network.nn
    nt = network.nt
    hk = xk[np+1:np+nn, :]

    # set optimizaiton solver and its attributes
    model = Model(()->Gurobi.Optimizer(Gurobi.Env()))
    set_silent(model)

    # define variables
    @variable(model, z[i=1:nn, k=1:nt])
    # @variable(model, z_lo[i, k] ≤ z[i=1:nn, k=1:nt] ≤ z_up[i, k])

    # nodal pressure variation constraint
    @constraint(model, [i=1:nn, k=1:nt; k!=nt], z[i, k+1] .- z[i, k] .≤ δmax)
    @constraint(model, [i=1:nn, k=1:nt; k!=nt], -δmax .≤ z[i, k+1] .- z[i, k])

    # objective function
    @objective(model, Min, 
        sum(sum(-λk[i, k]*z[i, k] + (γ/2)*(hk[i, k] - z[i, k])^2 for i ∈ collect(1:nn)) for k ∈ collect(1:nt))
        )

    # run optimization model
    optimize!(model)

    return value.(z)

end