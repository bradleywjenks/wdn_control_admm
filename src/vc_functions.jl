### valve pcontrol (VC) functions
using FLoops
using LinearAlgebra
using SparseArrays
using ProgressBars
using Printf

function solve_nlp_ipopt(network, data, obj_type; v_loc=nothing, y_loc=nothing, ω=nothing, f_data=nothing, autodiff=true, η_init=nothing, x_init=nothing, bi_dir=false)

    # unload network data
    elev = network.elev
    nexp = network.nexp
    d = network.d
    h0 = network.h0
    A10 = network.A10
    A12 = network.A12
    np, nn = size(A12)
    nt = network.nt
    azp_weights = data.azp_weights
    scc_weights = data.scc_weights

    # unload problem data
    q_lo = copy(data.Qmin)
    q_up = copy(data.Qmax)
    h_lo = copy(data.Hmin)
    h_up = copy(data.Hmax)
    η_lo = copy(data.ηmin)
    η_up = copy(data.ηmax)
    α_up = copy(data.αmax)
    α_lo = zeros(nn, nt)
    ρ = copy(data.ρ)
    umin = copy(data.umin)

    q_0 = x_init[1:np, :]
    h_0 = x_init[np+1:np+nn, :]
    η_0 = x_init[np+nn+1:2*np+nn, :]
    α_0 = x_init[2*np+nn+1:end, :]

    # define HW head loss model (nonlinear)
    r = determine_r(network.L, network.D, network.C, network.nexp, valve_idx=network.valve_idx)
    ϕ_ex(q, r, nexp) = r*q*abs.(q)^(nexp-1) # define head loss model
    ∇ϕ_ex(g, q, r, nexp) = g[1] =  nexp*r*abs.(q)^(nexp-1) # exact first derivative of head loss model
    ∇²ϕ_ex(H, q, r, nexp) = H[1, 1] = nexp*(nexp-1)*r*abs.(q)^(nexp-2)*ifelse(signbit(q), -1, 1) # exact second derivative of head loss model

    # define SCC sigmoid function (nonlinear)
    A = 1 ./ ((π/4).*network.D.^2)
    
    ψ_ex(q, A, s; ρ=ρ, umin=umin) = 1/(1+exp(-ρ*((s*q*A) - umin)))
    ∇ψ_ex(g, q, A, s; ρ=ρ, umin=umin) = g[1] = (s*ρ*A)*exp(-ρ*((s*q*A)-umin))*(1/(1+exp(-ρ*((s*q*A)-umin)))^2)
    ∇²ψ_ex(H, q, A, s; ρ=ρ, umin=umin) = H[1, 1] = (s*ρ*A)^2*((2*exp(-2*ρ*((s*q*A)-umin))*((1+exp(-ρ*((s*q*A)-umin)))^-3) - exp(-ρ*((s*q*A)-umin))*((1+exp(-ρ*((s*q*A)-umin)))^-2)))

    # define bilinear constraint for pressure control valve operation
    function β_ex(x...)
        return x[1]*x[2]
    end
    function ∇β_ex(g, x...)
        g[1] = x[2]
        g[2] = x[1]
        return
    end
    function ∇²β_ex(H, x...)
        H[1,1] = 0
        H[2,1] = 1
        H[2,2] = 0
    end


    # find junction and source nodes
    nodes_map = Dict(i=> findall(network.A12[i, :].!=0) for i in 1:network.np)
    sources_map = Dict(i=> findall(network.A10[i, :].!=0) for i in 1:network.np)


    # set optimizaiton solver and its attributes
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 1000)
    set_optimizer_attribute(model, "warm_start_init_point", "yes")
    set_optimizer_attribute(model, "linear_solver", "ma57")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "mu_oracle", "quality-function")
    set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
    # set_optimizer_attribute(model, "start_with_resto", "yes")
    # set_optimizer_attribute(model, "expect_infeasible_problem", "yes")
    set_optimizer_attribute(model, "tol", 1e-3)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-4)
    set_optimizer_attribute(model, "fast_step_computation", "yes")
    # set_optimizer_attribute(model, "hessian_approximation", "exact")
    # set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    # set_optimizer_attribute(model, "derivative_test", "first-order")
    # set_optimizer_attribute(model, "derivative_test", "second-order")
    set_optimizer_attribute(model, "print_level", 5)
    if obj_type == "scc" || obj_type == "azp-scc" || obj_type == "feasibility"
        set_optimizer_attribute(model, "start_with_resto", "yes")
        set_optimizer_attribute(model, "expect_infeasible_problem", "yes")
    end

    # define variables
    @variable(model, q_lo[i, k] ≤ q[i=1:np, k=1:nt] ≤ q_up[i, k])
    @variable(model, h_lo[i, k] ≤ h[i=1:nn, k=1:nt] ≤ h_up[i, k])
    @variable(model, η_lo[i, k] ≤ η[i=1:np, k=1:nt] ≤ η_up[i, k])
    @variable(model, α_lo[i, k] ≤ α[i=1:nn, k=1:nt] ≤ α_up[i, k])
    @variable(model, ψ⁺[i=1:np, k=1:nt])
    @variable(model, ψ⁻[i=1:np, k=1:nt])


    # hydraulic constraints
    if !autodiff
        register(model, :ϕ_ex, 3, ϕ_ex, ∇ϕ_ex, ∇²ϕ_ex)
        @NLconstraint(model, [i=1:np, k=1:nt], ϕ_ex(q[i, k], r[i], nexp[i]) + sum(A12[i, j]*h[j, k] for j ∈ nodes_map[i]) + sum(A10[i, j]*h0[j, k] for j ∈ sources_map[i]) + η[i, k] == 0)
    else
        @NLconstraint(model, [i=1:np, k=1:nt], r[i]*q[i, k]*abs(q[i, k])^(nexp[i]-1) + sum(A12[i, j]*h[j, k] for j ∈ nodes_map[i]) + sum(A10[i, j]*h0[j, k] for j ∈ sources_map[i]) + η[i, k] == 0)
    end
    @constraint(model, A12'*q - α .== d)

    # bilinear constraint for dbv control direction
    if bi_dir
        ϵ = -0.1
        if !autodiff
            register(model, :β_ex, 2, β_ex, ∇β_ex, ∇²β_ex)
            @NLconstraint(model, [i=1:np, k=1:nt; i in v_loc], ϵ ≤ β_ex(q[i, k], η[i, k]))
        else
            @NLconstraint(model, [i=1:np, k=1:nt; i in v_loc], ϵ ≤ η[i, k]*q[i, k])
        end
    end

    # auxiliary variables for scc sigmoid functions
    if !autodiff
        register(model, :ψ_ex, 3, ψ_ex, ∇ψ_ex, ∇²ψ_ex)
        @NLconstraint(model, [i=1:np, k=1:nt], ψ⁺[i, k] == ψ_ex(q[i, k]/1000, A[i], 1))
        @NLconstraint(model, [i=1:np, k=1:nt], ψ⁻[i, k] == ψ_ex(q[i, k]/1000, A[i], -1))
    else
        @NLconstraint(model, [i=1:np, k=1:nt], ψ⁺[i, k] ==  (1+exp(-ρ*((q[i, k]/1000*A[i]) - umin)))^-1)
        @NLconstraint(model, [i=1:np, k=1:nt], ψ⁻[i, k] ==  (1+exp(-ρ*(-(q[i, k]/1000*A[i]) - umin)))^-1)
    end

    # objective function
    if obj_type == "azp"
        @objective(model, Min, (1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt))
    elseif obj_type == "scc"
        @objective(model, Min, (1/nt)*sum(sum(-scc_weights[j]*(ψ⁺[j, k] + ψ⁻[j, k]) for j ∈ 1:np) for k ∈ 1:nt)) 
    elseif obj_type == "azp-scc"
        f1a=f_data.f1a
        f1b=f_data.f1b
        f2a=f_data.f2a
        f2b=f_data.f2b
        @objective(model, Min, 
            (1-ω)*((((1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt)) - f1a) / (f1b - f1a)) + ω*((((1/nt)*sum(sum(-scc_weights[j]*(ψ⁺[j, k] + ψ⁻[j, k]) for j ∈ 1:np) for k ∈ 1:nt)) - f2a) / (f2b - f2a)))
    elseif obj_type == "feasibility"
        # no objective function
        @objective(model, Min, (1/(np*nt))*sum(sum((η[j, k] - η_0[j, k])^2 for j ∈ 1:np) for k ∈ 1:nt))
    end

    # set starting values
    ψ⁺_0 = zeros(np, nt)
    ψ⁻_0= zeros(np, nt)
    for k ∈ 1:nt
        for i ∈ 1:np
            ψ⁺_0[i, k] = ψ_ex(q_0[i, k]/1000, A[i], 1)
            ψ⁻_0[i, k] = ψ_ex(q_0[i, k]/1000, A[i], -1)
        end
    end
    set_start_value.(ψ⁺, ψ⁺_0)
    set_start_value.(ψ⁻, ψ⁻_0)
    set_start_value.(q, q_0)
    set_start_value.(h, h_0)
    set_start_value.(η, η_0)

    # run optimization model
    optimize!(model)
    solution_summary(model)
    term_status = termination_status(model)

    accepted_status = [LOCALLY_SOLVED; ALMOST_LOCALLY_SOLVED; OPTIMAL; ALMOST_OPTIMAL]

    if term_status in accepted_status
        q_sol = value.(q)
        h_sol = value.(h)
        η_sol = value.(η)
        ψ⁺_sol = value.(ψ⁺)
        ψ⁻_sol = value.(ψ⁻)
        α_sol = value.(α)
        obj_sol = objective_value(model)
        cpu_sol = solve_time(model)
        x_sol = reshape(vcat(q_sol, h_sol, η_sol, α_sol), nt*(2*np+2*nn), 1)

    else
        q_sol = NaN
        h_sol = NaN
        η_sol = NaN
        ψ⁺sol = NaN
        ψ⁻sol = NaN
        α_sol = NaN
        cpu_sol = NaN
        obj_sol = Inf
        x_sol = repeat([NaN], nt*(2*np+2*nn), 1)
    end


    if obj_type == "feasibility" && term_status in accepted_status
        return value.(q), value.(h), value.(η)
    else
        return x_sol, obj_sol, cpu_sol
    end

end





function solve_nlp_sfscp(network, probdata, obj_type; v_loc=nothing, y_loc=nothing, ω=nothing, f_data=nothing, η_init=nothing, max_iter=100, ϵ_tol=1e-3)

    time_start = time()
    
    # unload network data
    elev = network.elev
    nexp = network.nexp
    d = network.d
    h0 = network.h0
    A10 = network.A10
    A12 = network.A12
    np, nn = size(A12)
    nt = network.nt
    azp_weights = probdata.azp_weights
    scc_weights = probdata.scc_weights
    r = determine_r(network.L, network.D, network.C, network.nexp, valve_idx=network.valve_idx)
    A = 1 ./ ((π/4).*network.D.^2)

    # unload problem data
    q_lo = copy(probdata.Qmin)
    q_up = copy(probdata.Qmax)
    h_lo = copy(probdata.Hmin)
    h_up = copy(probdata.Hmax)
    η_lo = copy(probdata.ηmin)
    η_up = copy(probdata.ηmax)
    α_up = copy(probdata.αmax)
    α_lo = zeros(nn, nt)
    ρ = copy(probdata.ρ)
    umin = copy(probdata.umin)

    # define functions 
    ϕ(q, r, nexp) = r*q*abs.(q)^(nexp-1) # HW head loss model
    ∇ϕ_q(q, r, nexp) = nexp*r*abs.(q)^(nexp-1) # first derivative of head loss model
    ψ(q, A, s; ρ=ρ, umin=umin) = 1/(1+exp(-ρ*((s*q*A) - umin)))
    ∇ψ_q(q, A, s; ρ=ρ, umin=umin) = (s*ρ*A)*exp(-ρ*((s*q*A)-umin))*(1/(1+exp(-ρ*((s*q*A)-umin)))^2)


    # find junction and source nodes
    nodes_map = Dict(i => findall(A12[i, :] .!= 0) for i in 1:np)
    sources_map = Dict(i => findall(A10[i, :] .!= 0) for i in 1:np)

    # modify η and α bounds based on z_loc and y_loc, respectively
    A13 = spzeros(np)
    # for (i, v) ∈ enumerate(z_loc)
    #     if v ≤ np
    #         η_lo[v, :] .= 0 
    #         q_lo[v, :] .= 0
    #         A13[v] = 1
    #     else
    #         η_up[v-np, :] .= 0
    #         q_up[v-np, :] .= 0
    #         A13[v-np] = 1
    #         z_loc[i] = v-np
    #     end
    # end
    A13[z_loc] .=1
    η_lo[setdiff(1:np, z_loc), :] .= 0
    η_up[setdiff(1:np, z_loc), :] .= 0

    if nf >0 
        α_up[setdiff(1:nn, y_loc), :] .= 0
    else
        α_up .= 0
    end

    # compute feasible starting point data
    x_k = make_starting_point(network, probdata, obj_type, q_lo, q_up, h_lo, h_up, η_lo, η_up, η_init, z_loc, y_loc, A13)
    q_k = x_k[1:np, :]
    h_k = x_k[np+1:np+nn, :]
    η_k = x_k[np+nn+1:2*np+nn, :]
    α_k = x_k[2*np+nn+1:end, :]

    a_k = [ϕ(q_k[i, t], r[i], nexp[i])*(1-nexp[i]) for i in 1:np, t in 1:nt]
    b_k = [∇ϕ_q(q_k[i, t], r[i], nexp[i]) for i in 1:np, t in 1:nt].*-1

    # check feasibility of starting point
    feasible = is_feasible(q_lo, q_up, h_lo, h_up, η_lo, η_up, q_k, h_k, η_k)
    if !feasible
        @info "Starting point is not feasible"
        x_k = repeat([NaN], nt*(2*np+2*nn), 1)
        obj_k = Inf
        cpu_time = NaN
        return x_k, obj_k, cpu_time
    end

    # compute initial objective function value
    obj_k = objective_function(network, probdata, obj_type, q_k, h_k, ω, f_data)

    # initialize results matrices and add initial data
    η_hist = []
    push!(η_hist, η_k)
    α_hist = []
    push!(α_hist, α_k)
    obj_hist = []
    push!(obj_hist, obj_k)
    @printf("%d \t %f \t - \n", 0, obj_k)

    # build model
    model = Model(Gurobi.Optimizer)
    #set_optimizer_attribute(model,"Method", 2)
    # set_optimizer_attribute(model,"Presolve", 0)
    # set_optimizer_attribute(model,"Crossover", 0)
    #set_optimizer_attribute(model,"NumericFocus", 3)
    # set_optimizer_attribute(model,"NonConvex", 2)
    set_silent(model)
    
    # define variables
    @variable(model, q_lo[i, k] ≤ q[i=1:np, k=1:nt] ≤ q_up[i, k])
    @variable(model, h_lo[i, k] ≤ h[i=1:nn, k=1:nt] ≤ h_up[i, k])
    @variable(model, η_lo[i, k] ≤ η[i=1:np, k=1:nt] ≤ η_up[i, k])
    @variable(model, α_lo[i, k] ≤ α[i=1:nn, k=1:nt] ≤ α_up[i, k])
    @variable(model, θ[i=1:np, k=1:nt]) # auxiliary variable for linearized part of energy equation
    @variable(model, ψ⁺[i=1:np, k=1:nt])
    @variable(model, ψ⁻[i=1:np, k=1:nt])

    # set objective function
    if obj_type=="azp"
        @objective(model, Min, (1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt))
    elseif obj_type =="scc"
        @objective(model, Min, (1/nt)*sum(sum(-scc_weights[i]*(ψ⁺[i, k] + ψ⁻[i, k]) for i ∈ 1:np) for k ∈ 1:nt))
    elseif obj_type == "azp-scc"
        f1a = f_data.f1a
        f1b = f_data.f1b
        f2a = f_data.f2a
        f2b = f_data.f2b
        @objective(model, Min, 
            (1-ω)*((((1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt)) - f1a) / (f1b - f1a)) + ω*((((1/nt)*sum(sum(-scc_weights[j]*(ψ⁺[j, k] + ψ⁻[j, k]) for j ∈ 1:np) for k ∈ 1:nt)) - f2a) / (f2b - f2a)))
    end

    # linear energy and mass constraint
    @constraint(model, θ + A12*h + A10*h0 + η .== 0)
    @constraint(model, A12'*q - α .== d)

    # linearized hydraulic constraint
    @constraint(model, linearised_hyd[i=1:np, k=1:nt], θ[i, k] + b_k[i, k]*q[i, k] == a_k[i, k])

    # # bilinear control valve constraint
    ϵ_bi = -0.01 # constraint violation tolerance
    @constraint(model, linearised_bi[i=1:np, k=1:nt; i in z_loc], η_k[i, k]*q[i, k] + q_k[i, k]*η[i, k] ≥ ϵ_bi + q_k[i, k]*η_k[i, k])
    # @constraint(model, linearised_bi[i=1:np, k=1:nt], η[i, k]*q[i, k] ≥ ϵ_bi)

    # auxiliary variables for scc sigmoid functions
    @constraint(model, linearised_ψ⁺[i=1:np, k=1:nt], ψ⁺[i, k] - ∇ψ_q(q_k[i, k]/1000, A[i], 1)*q[i, k] == ψ(q_k[i, k]/1000, A[i], 1) - ∇ψ_q(q_k[i, k]/1000, A[i], 1)*q_k[i, k])
    @constraint(model, linearised_ψ⁻[i=1:np, k=1:nt], ψ⁻[i, k] - ∇ψ_q(q_k[i, k]/1000, A[i], -1)*q[i, k] == ψ(q_k[i, k]/1000, A[i], -1) - ∇ψ_q(q_k[i, k]/1000, A[i], -1)*q_k[i, k])



    # perform strictly feasible sequential convex programming (sfscp) algorithm
    iter = 1:max_iter
    for kk ∈ iter

        # Step 1: compute line search step
        optimize!(model)
        η_t = value.(η)
        α_t = value.(α)

        # Step 3: acceptance of trial point and line search
        q_t, h_t, _, _ = hydraulic_simulation(network, probdata.Qmax, η_t, A13, α_t)
        if !all()
            !all(q_t != Inf) || !all(q_t != NaN)
            obj_t == Inf
            feasible = false
        else
            obj_t = objective_function(network, probdata, obj_type, q_t, h_t, ω, f_data)
            feasible = is_feasible(q_lo, q_up, h_lo, h_up, η_lo, η_up, q_t, h_t, η_t)
        end
        ∂η = η_t - η_k
        ∂α = α_t - α_k

        β = 1
        while  !all(q_t != Inf) || !all(q_t != NaN) || obj_t - obj_k ≥ 0 || !feasible
            β = 0.5*β
            η_t = η_k + β*∂η
            α_t = α_k + β*∂α
            q_t, h_t, _, _ = hydraulic_simulation(network, probdata.Qmax, η_t, A13, α_t)
            if !all(q_t != Inf) || !all(q_t != NaN)
                obj_t == Inf
                feasible = false
            else
                obj_t = objective_function(network, probdata, obj_type, q_t, h_t, ω, f_data)
                feasible = is_feasible(q_lo, q_up, h_lo, h_up, η_lo, η_up, q_t, h_t, η_t)
            end
            if norm(β) < 1e-6
                break
            end
        end

        if feasible
            # success!
            obj_old = obj_k
            obj_k = obj_t
            η_k = η_t
            α_k = α_t
            h_k = h_t
            q_k = q_t

            for k ∈ collect(1:nt)
                for i ∈ collect(1:np)
                    # update model parameters
                    a_k = ϕ(q_k[i, k], r[i], nexp[i])*(1-nexp[i])
                    b_k = ∇ϕ_q(q_k[i, k], r[i], nexp[i]).*-1

                    # linearised energy conservation constraint
                    set_normalized_coefficient(linearised_hyd[i, k], q[i, k], b_k)
                    set_normalized_rhs(linearised_hyd[i, k], a_k)

                    if i ∈ z_loc
                        # linearised bilinear control valve constraint
                        set_normalized_coefficient(linearised_bi[i, k], q[i, k], η_k[i, k])
                        set_normalized_coefficient(linearised_bi[i, k], η[i, k], q_k[i, k])
                        set_normalized_rhs(linearised_bi[i, k], ϵ_bi + q_k[i, k]*η_k[i, k])
                    end

                    # linearised ψ⁺ inequality constraint
                    set_normalized_coefficient(linearised_ψ⁺[i, k], q[i, k], -∇ψ_q(q_k[i, k]/1000, A[i], 1))
                    set_normalized_rhs(linearised_ψ⁺[i, k], ψ(q_k[i, k]/1000, A[i], 1) - ∇ψ_q(q_k[i, k]/1000, A[i], 1)*q_k[i, k])

                    # linearised ψ⁻ inequality constraint
                    set_normalized_coefficient(linearised_ψ⁻[i, k], q[i, k], -∇ψ_q(q_k[i, k]/1000, A[i], -1))
                    set_normalized_rhs(linearised_ψ⁻[i, k], ψ(q_k[i, k]/1000, A[i], -1) - ∇ψ_q(q_k[i, k]/1000, A[i], -1)*q_k[i, k])
                end
            end
            Ki = abs(obj_old-obj_k)/abs(obj_old)
        else
            # do nothing and stop
            Ki = 0
        end
        push!(η_hist, η_k)
        push!(α_hist, α_k)
        push!(obj_hist, obj_k)

        @printf("%d \t %f \t %f \t \t \n", kk, obj_k, Ki) 
        if Ki ≤ ϵ_tol
            break
        end
    end

    cpu_time = time()-time_start
    x_k= reshape(vcat(q_k, h_k, η_k, α_k), nt*(2*np+2*nn), 1)

    return x_k, obj_k, cpu_time

end


"""
    feasibility_restoration
    - restores feasibility (if needed) of inputted η starting values
"""

function make_starting_point(network, data, obj_type, η, v, y, A13, bi_dir)

    # unload problem data
    q_lo = copy(data.Qmin)
    q_up = copy(data.Qmax)
    h_lo = copy(data.Hmin)
    h_up = copy(data.Hmax)
    η_lo = copy(data.ηmin)
    η_up = copy(data.ηmax)

    if η === nothing
        q_init, h_init, _, _ = hydraulic_simulation(network, data)
        x_init = vcat(q_init, h_init, zeros(network.np, network.nt), zeros(network.nn, network.nt))
        feas = true

    else
        #covex relaxation η solution
        η[setdiff(1:network.np, v), :] .= 0
        # η = η .* 0.5
        q, h, _, _ = hydraulic_simulation(network, data.Qmax, η, A13)

        # check feasibility
        feasible = is_feasible(q_lo, q_up, h_lo, h_up, η_lo, η_up, q, h, η)

        # feasibility restoration problem
        if !feasible
            # x_init for nlp problem
            x_init = vcat(q, h, η, zeros(network.nn, network.nt))

            # run optimization model
            try 
                obj_feas = "feasibility"
                q_feas, h_feas, η_feas = solve_nlp_ipopt(network, data, obj_feas; v_loc=v, y_loc=y, x_init=x_init, bi_dir=bi_dir)

                # confirm hydraulic feasibility
                feasible = is_feasible(q_lo, q_up, h_lo, h_up, η_lo, η_up, q_feas, h_feas, η_feas)

                if feasible
                    x_init = vcat(q_feas, h_feas, η_feas, zeros(network.nn, network.nt))
                    feas = true
                else
                    q_init, h_init, _, _ = hydraulic_simulation(network, data)
                    x_init = vcat(q_init, h_init, zeros(network.np, network.nt), zeros(network.nn, network.nt))
                    feas = false
                end

            catch e
                q_init, h_init, _, _ = hydraulic_simulation(network, data)
                x_init = vcat(q_init, h_init, zeros(network.np, network.nt), zeros(network.nn, network.nt))
                feas = false
            end
            
        else
            # inputted η is hydrualically feasible
            x_init = vcat(q, h, η, zeros(network.nn, network.nt))
            feas = true
        end
    end

    return x_init, feas
    
end



function is_feasible(q_lo, q_up, h_lo, h_up, η_lo, η_up, q, h, η)

    η_bounds = all((η_lo .- 1e-03) .≤ η .≤ (η_up .+ 1e-03))
    η_dir = all(-0.105 .≤ q.*η)
    h_bounds = all(h_lo .- 1e-03 .≤ h .≤ h_up .+ 1e-03)
    q_bounds = all((q_lo .- 1e-03) .≤ q .≤ (q_up .+ 1e-03))
    all_feas = all((η_bounds, η_dir, h_bounds, q_bounds))

    return all_feas

end



function fix_valve_locs(valve_visited, nv, nf)
    if nv > 0 && nf > 0
        v_loc = valve_visited[1:nv]
        y_loc = valve_visited[nv+1:end]
    elseif nv == 0 && nf > 0
        v_loc = []
        y_loc = valve_visited
    elseif nv > 0 && nf == 0
        v_loc = valve_visited
        y_loc = []
    end

    return v_loc, y_loc
end



function objective_function(network, probdata, obj_type, q_t, h_t, ω, f_data)

    # unload network data
    elev = network.elev
    np, nn = size(network.A12)
    nt = network.nt
    azp_weights = probdata.azp_weights
    scc_weights = probdata.scc_weights
    ρ = probdata.ρ
    umin = probdata.umin
    D = network.D
    A = 1 ./ ((π/4).*D.^2 .*1000)
    
    # compute objective function value
    if obj_type == "azp"
        f = (1/nt)*sum(sum(azp_weights[i]*(h_t[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt)
    elseif obj_type == "scc"
        f = (1/nt)*sum(sum(-scc_weights[j]*(1/(1+exp(-ρ*((q_t[j, k]*A[j]) - umin))) + 1/(1+exp(-ρ*((-1*q_t[j, k]*A[j]) - umin)))) for j ∈ 1:np) for k ∈ 1:nt)
    elseif obj_type == "azp_scc"
        f1a = f_data.f1a
        f1b = f_data.f1b
        f2a = f_data.f2a
        f2b = f_data.f2b

        f1x = (1/nt)*sum(sum(azp_weights[i]*(h_t[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt)
        f2x = (1/nt)*sum(sum(-scc_weights[j]*(1/(1+exp(-ρ*((q_t[j, k]*A[j]) - umin))) + 1/(1+exp(-ρ*((-1*q_t[j, k]*A[j]) - umin)))) for j ∈ 1:np) for k ∈ 1:nt)

        f = (1-ω)*((f1x - f1a) / (f1b - f1a)) + ω*((f2x - f2a) / (f2b - f2a))
    end

    return f

end