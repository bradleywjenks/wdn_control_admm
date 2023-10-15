### load dependencies
using JuMP
using Ipopt
using LaTeXStrings
using Statistics
using Plots
using FileIO
using JLD2

include("src/admm_functions.jl")


#################### PRELIMINARIES ####################

### input problem parameters ###
begin
    # net_name = "bwfl_2022_05_hw"
    # net_name = "L_town"
    net_name = "modena"

    n_v = 3
    n_f = 4

    pv_type = "range" # pv_type = "variation"; pv_type = "variability"; pv_type = "range"; pv_type = "none"
    pv_active = true
    δmax = 10
    δviol = 1.24 # allowed constraint violation on the basis of ADMM results

    obj_type = "azp-scc"

    resto = false
end


### load problem data ###
begin
    # net_name = "bwfl_2022_05_hw"
    data = load("data/problem_data/"*net_name*"_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2")
end


### centralised problem ### 
cpu_time = @elapsed begin

    nt = data["nt"]
    # time_steps = collect(1:3:nt)
    time_steps = collect(1:nt)

     # unload data
    elev = data["elev"]
    nexp = data["nexp"]
    A10 = data["A10"]
    A12 = data["A12"]
    np, nn = size(A12)
    nt = length(time_steps)
    d = data["d"][:, time_steps]
    h0 = data["h0"][:, time_steps]
    r = data["r"]
    A = 1 ./ ((π/4).*data["D"].^2)
    azp_weights = data["azp_weights"]
    scc_weights = data["scc_weights"]
    q_lo = data["Qmin"][:, time_steps]
    q_up = data["Qmax"][:, time_steps]
    h_lo = data["Hmin"][:, time_steps]
    h_up = data["Hmax"][:, time_steps]
    η_lo = data["ηmin"][:, time_steps]
    η_up = data["ηmax"][:, time_steps]
    α_up = data["αmax"][:, time_steps]
    α_lo = zeros(nn, nt)
    y_loc = data["y_loc"]
    v_loc = data["v_loc"]
    # scc_time = data["scc_time"]
    scc_time = collect(7:8)
    ρ = data["ρ"]
    umin = data["umin"]

    α_up[y_loc, scc_time] .= 25

    # define nonlinear SCC objective function
    ψ_ex(q, A, s; ρ=ρ, umin=umin) = 1/(1+exp(-ρ*((s*q*A) - umin)))

    # find junction and source nodes
    nodes_map = Dict(i=> findall(A12[i, :].!=0) for i in 1:np)
    links_map = Dict(i=> findall(A12[:, i].!=0) for i in 1:nn)
    sources_map = Dict(i=> findall(A10[i, :].!=0) for i in 1:np)

    # set optimizaiton solver and its attributes
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 3000)
    set_optimizer_attribute(model, "warm_start_init_point", "yes")
    set_optimizer_attribute(model, "linear_solver", "ma57")
    # set_optimizer_attribute(model, "ma57_pivtol", 1e-8)
    # set_optimizer_attribute(model, "ma57_pre_alloc", 10.0)
    # set_optimizer_attribute(model, "ma57_automatic_scaling", "yes")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "mu_oracle", "quality-function")
    set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
    # set_optimizer_attribute(model, "tol", 1e-6)
    # set_optimizer_attribute(model, "constr_viol_tol", 1e-9)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-2)
    # set_optimizer_attribute(model, "fast_step_computation", "yes")
    # set_optimizer_attribute(model, "hessian_approximation", "exact")
    # set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    # set_optimizer_attribute(model, "derivative_test", "first-order")
    # set_optimizer_attribute(model, "derivative_test", "second-order")
    set_optimizer_attribute(model, "print_level", 5)
    if resto
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

    # hydraulic conservation constraints
    reg = 1e-08
    @NLconstraint(model, [i=1:np, k=1:nt], r[i]*(q[i, k]+reg)*abs(q[i, k]+reg)^(nexp[i]-1) + sum(A12[i, j]*h[j, k] for j ∈ nodes_map[i]) + sum(A10[i, j]*h0[j, k] for j ∈ sources_map[i]) + η[i, k] == 0)
    @constraint(model, A12'*q - α .== d)

    # bilinear constraint for dbv control direction
    ϵ_bi = 0
    @NLconstraint(model, [i=1:np, k=1:nt; i in v_loc], ϵ_bi ≤ η[i, k]*q[i, k])

    # auxiliary variables for scc sigmoid functions
    @NLconstraint(model, [i=1:np, k=1:nt], ψ⁺[i, k] == (1+exp(-ρ*((q[i, k]/1000*A[i]) - umin)))^-1)
    @NLconstraint(model, [i=1:np, k=1:nt], ψ⁻[i, k] == (1+exp(-ρ*(-(q[i, k]/1000*A[i]) - umin)))^-1)

    # pressure variation (pv) constraint
    if pv_active
        if pv_type == "variation"
            @constraint(model, [i=1:nn, k=1:nt; k!=nt], h[i, k+1] .- h[i, k] .≤ (δmax + δviol))
            @constraint(model, [i=1:nn, k=1:nt; k!=nt], -1*(δmax + δviol) .≤ h[i, k+1] .- h[i, k])
    
        elseif pv_type == "variability"
            reg = 1e-12
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
            @constraint(model, [i=1:nn], vec(h[i, :])'*A*vec(h[i, :]) .≤ (δmax + δviol)^2)
     
        elseif pv_type == "range"
            @variable(model, l[i=1:nn])
            @variable(model, u[i=1:nn])
            @constraint(model, [i=1:nn, k=1:nt], h[i, k] ≤ u[i])
            @constraint(model, [i=1:nn, k=1:nt], l[i] ≤ h[i, k])
            @constraint(model, [i=1:nn], u[i] - l[i] .≤ (δmax + δviol))
    
        elseif pv_type == "none"
            # nothing
        end
    end

    # objective function
    if obj_type == "azp"
        @objective(model, Min, (1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt))
    elseif obj_type == "scc"
        @objective(model, Min, (1/nt)*sum(sum(-scc_weights[j]*(ψ⁺[j, k] + ψ⁻[j, k]) for j ∈ 1:np) for k ∈ 1:nt)) 
    elseif obj_type == "azp-scc"

        azp_time = collect(1:nt)
        deleteat!(azp_time, scc_time)

        @objective(model, Min,
            sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ azp_time) + sum(sum(-scc_weights[j]*(ψ⁺[j, k] + ψ⁻[j, k]) for j ∈ 1:np) for k ∈ scc_time))

    end

    # unload and set starting values
    # q_init = data["q_init"]
    # h_init = data["h_init"]
    q_init = data["q_init"][:, time_steps]
    h_init = data["h_init"][:, time_steps]
    ψ⁺_init = zeros(np, nt)
    ψ⁻_init = zeros(np, nt)
    for k ∈ collect(1:nt)
        for j ∈ collect(1:np)
            ψ⁺_init[j, k] =  (1+exp(-ρ*((q_init[j, k]/1000*A[j]) - umin)))^-1
            ψ⁻_init[j, k] = (1+exp(-ρ*(-(q_init[j, k]/1000*A[j]) - umin)))^-1
        end
    end
    set_start_value.(ψ⁺, ψ⁺_init)
    set_start_value.(ψ⁻, ψ⁻_init)
    set_start_value.(q, q_init)
    set_start_value.(h, h_init)

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
        α_sol = value.(α)
        ψ⁺_sol = value.(ψ⁺)
        ψ⁻_sol = value.(ψ⁻)
        obj_sol = objective_value(model)
        cpu_sol = solve_time(model)
        x_sol = vcat(q_sol, h_sol, η_sol, α_sol)

    else
        q_sol = NaN
        h_sol = NaN
        η_sol = NaN
        ψ⁺_sol = NaN
        ψ⁻_sol = NaN
        α_sol = NaN
        cpu_sol = NaN
        obj_sol = Inf
        x_sol = NaN
    end
end


### save data ###

begin
    @save "data/centralised_results/"*net_name*"_"*pv_type*"_"*string(δmax)*".jld2" x_sol obj_sol cpu_time
end


### load data ###
begin
    @load "data/centralised_results/"*net_name*"_"*pv_type*"_inf.jld2" x_sol obj_sol cpu_time
end



### plotting code ###

### plot objective function (time series) ### 
begin
    qk = x_sol[1:np, :]
    hk = x_sol[np+1:np+nn, :]
    A = 1 ./ ((π/4).*data["D"].^2)
    f_azp = zeros(nt)
    f_scc = zeros(nt)
    f_val = zeros(nt)
    for k ∈ 1:nt
        f_azp[k] = sum(data["azp_weights"][i]*(hk[i, k] - data["elev"][i]) for i ∈ 1:nn)
        f_scc[k] = sum(data["scc_weights"][j]*((1+exp(-ρ*((qk[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ*(-(qk[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
        if k ∈ scc_time
            f_val[k] = f_scc[k]*-1
        else
            f_val[k] = f_azp[k]
        end
    end

    plot_azp = plot()
    plot_azp = plot!(collect(1:nt), f_azp, c=:red3, seriestype=:line, linewidth=2, linestyle=:solid, label="")
    plot_azp = vspan!([scc_time[1], scc_time[end]], c=:black, alpha = 0.1, label = "")
    plot_azp = plot!(xlabel="", ylabel="AZP [m]",  xlims=(0, 24), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, legendborder=:false, legend=:best, bottom_margin=2*Plots.mm, size=(600, 550))
    # plot_azp = plot!(xlabel="", ylabel="AZP [m]",  xlims=(0, 96), xticks=(0:24:96), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=16, legendfont=14, legendborder=:false, legend=:none, bottom_margin=2*Plots.mm, fontfamily="Computer Modern", size=(600, 600))
    plot_scc = plot()
    plot_scc = plot!(collect(1:nt), f_scc, c=:red3, seriestype=:line, linewidth=2, linestyle=:solid, label="")
    plot_scc = vspan!([scc_time[1], scc_time[end]], c=:black, alpha = 0.1, label = "")
    plot_scc = plot!(xlabel="Time step", ylabel=L"SCC $[\%]$", xlims=(0, 24), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, legend=:none, size=(600, 550))
    # plot_scc = plot!(xlabel="Time step", ylabel="SCC [%]", xlims=(0, 96), xticks=(0:24:96), xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16, legendfont=14, legend=:none, fontfamily="Computer Modern", size=(600, 600))
    plot(plot_azp, plot_scc, layout=(2, 1))

    # ylims=(30, 55)
    # ylims=(0, 50)
    # xticks=(0:24:96)
    # size=(425, 500)
end


r = maximum(hk, dims=2) - minimum(hk, dims=2)

maximum(r)

histogram(r, bins=50)

sum(f_val)

f_val