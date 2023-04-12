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
    # net_name = "modena"
    net_name = "L_town"
    # net_name = "bwkw_mod"

    # make_data = true
    make_data = false
    # bv_open = true
    bv_open = false

    n_v = 3
    n_f = 4
    αmax = 25
    δmax = 20
    umin = 0.2
    ρ = 50
    # pmin = 10 # for bwkw network
    obj_type = "azp-scc"
    pv_type = "variability" # pv_type = "variation"; pv_type = "variability"; pv_type = "range"; pv_type = "none"
    pv_active = true

    # scc_time = collect(38:42) # bwfl (peak)
    scc_time = collect(28:32) # bwkw (peak)
    # scc_time = collect(12:16) # bwfl (min)
    # scc_time = collect(7:8) # modena (peak)
    # scc_time = collect(3:4) # modena (min)
    # scc_time = []

    resto = false
end

### make network and problem data ###
begin
    if make_data
        using OpWater
        # load network data
        if net_name == "bwfl_2022_05_hw"
            load_name = "bwfl_2022_05/hw"
        else
            load_name = net_name
        end
        network = load_network(load_name, afv_idx=false, dbv_idx=false, pcv_idx=false, bv_open=bv_open)

        # make optimization parameters
        opt_params = make_prob_data(network; αmax_mul=αmax, umin=umin, ρ=ρ)
        q_init, h_init, err, iter = hydraulic_simulation(network, opt_params)
        S = (π * (network.D) .^ 2) / 4
        v0 = q_init ./ (1000 * S) 
        max_v = ceil(maximum(abs.(v0)))
        opt_params.Qmin , opt_params.Qmax, opt_params.umax = q_bounds_from_u(network, q_init; max_v=max_v)

        # load pcv and afv locations
        @load "data/single_objective_results/"*net_name*"_azp_random_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2" sol_best
        v_loc = sol_best.v
        @load "data/single_objective_results/"*net_name*"_scc_random_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2" sol_best
        y_loc = sol_best.y

        # save problem data
        make_object_data(net_name, network, opt_params, v_loc, y_loc)
    end
end


### load problem data ###
begin
    # net_name = "bwfl_2022_05_hw"
    data = load("data/problem_data/"*net_name*"_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2")
end


### monolithic problem ### 
cpu_time = @elapsed begin

     # unload data
    elev = data["elev"]
    nexp = data["nexp"]
    d = data["d"]
    h0 = data["h0"]
    A10 = data["A10"]
    A12 = data["A12"]
    np, nn = size(A12)
    nt = data["nt"]
    r = data["r"]
    A = 1 ./ ((π/4).*data["D"].^2)
    azp_weights = data["azp_weights"]
    scc_weights = data["scc_weights"]
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
    set_optimizer_attribute(model, "constr_viol_tol", 2.5e-1)
    set_optimizer_attribute(model, "fast_step_computation", "yes")
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
    ϵ_bi = -0.05
    @NLconstraint(model, [i=1:np, k=1:nt; i in v_loc], ϵ_bi ≤ η[i, k]*q[i, k])

    # auxiliary variables for scc sigmoid functions
    @NLconstraint(model, [i=1:np, k=1:nt], ψ⁺[i, k] ==  (1+exp(-ρ*((q[i, k]/1000*A[i]) - umin)))^-1)
    @NLconstraint(model, [i=1:np, k=1:nt], ψ⁻[i, k] ==  (1+exp(-ρ*(-(q[i, k]/1000*A[i]) - umin)))^-1)

    # pressure variation (pv) constraint
    if pv_active
        if pv_type == "variation"
            @constraint(model, [i=1:nn, k=1:nt; k!=nt], h[i, k+1] .- h[i, k] .≤ δmax)
            @constraint(model, [i=1:nn, k=1:nt; k!=nt], -δmax .≤ h[i, k+1] .- h[i, k])
    
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
            @constraint(model, [i=1:nn], vec(h[i, :])'*A*vec(h[i, :]) .≤ δmax^2)
     
        elseif pv_type == "range"
            @variable(model, l[i=1:nn])
            @variable(model, u[i=1:nn])
            @constraint(model, [i=1:nn, k=1:nt], h[i, k] ≤ u[i])
            @constraint(model, [i=1:nn, k=1:nt], l[i] ≤ h[i, k])
            @constraint(model, [i=1:nn], u[i] - l[i] .≤ δmax)
    
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
        function f_ex(x...; w1=azp_weights, w2=scc_weights, elev=elev, nt=nt, np=np, nn=nn, scc_time=scc_time)
            x = [x...]
            x = reshape(x, nn+2*np, nt)
            h = x[1:nn, :]
            ψ⁺ = x[nn+1:nn+np, :]
            ψ⁻ = x[nn+np+1:end, :]

            azp_time = collect(1:nt)
            deleteat!(azp_time, scc_time)

            # azp objective function value
            f_azp = (1/length(azp_time))*sum(sum(w1[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ azp_time)
            # scc objective function value
            f_scc = (1/length(scc_time))*sum(sum(-w2[i]*(ψ⁺[i, k] + ψ⁻[i, k]) for i ∈ 1:np) for k ∈ scc_time)

            f = f_azp + f_scc

            return f
        end
        function ∇f_ex(g, x...; w1=opt_params.azp_weights, w2=opt_params.scc_weights, elev=network.elev, nt=network.nt, np=network.np, nn=network.nn, scc_time=scc_time)
            azp_time = collect(1:nt)
            deleteat!(azp_time, scc_time)

            g_azp = vcat(w1, zeros(2*np))
            g_scc = vcat(zeros(nn), -w2, -w2)

            g = []
            for k ∈ 1:nt
                if any(x->x==k, azp_time)
                    append!(g, g_azp)
                elseif any(x->x==k, scc_time)
                    append!(g, g_scc)
                end
            end

            return 
        end
        function ∇²f_ex(H, x...; w1=opt_params.azp_weights, w2=opt_params.scc_weights, elev=network.elev, nt=network.nt, np=network.np, nn=network.nn, scc_time=scc_time)
            # nothing since JuMP already populates zeros in Hessian matrix
            return
        end
        register(model, :f_ex, length(vec(h))+length(vec(ψ⁺))+length(vec(ψ⁻)), f_ex, ∇f_ex, ∇²f_ex)
        @objective(model, Min, f_ex(vcat(h, ψ⁺, ψ⁻)...))
    end

    # unload and set starting values
    q_init = data["q_init"]
    h_init = data["h_init"]
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
    @save "data/admm_results/"*net_name*pv_type*"_delta_"*string(δmax)*"_monolith.jld2" x_sol obj_sol cpu_time
end

### load data ###
begin
    @load "data/admm_results/"*net_name*pv_type*"_delta_"*string(δmax)*"_monolith.jld2" x_sol obj_sol cpu_time
end


### plotting code ###

### plot objective function (time series) ### 
begin
    qk = x_sol[1:np, :]
    hk = x_sol[np+1:np+nn, :]
    A = 1 ./ ((π/4).*data["D"].^2)
    f_azp = zeros(nt)
    f_scc = zeros(nt)
    for k ∈ 1:nt
        f_azp[k] = sum(data["azp_weights"][i]*(hk[i, k] - data["elev"][i]) for i ∈ 1:nn)
        f_scc[k] = sum(data["scc_weights"][j]*((1+exp(-ρ*((qk[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ*(-(qk[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
    end

    plot_azp = plot()
    plot_azp = plot!(collect(1:nt), f_azp, c=:red3, seriestype=:line, linewidth=2, linestyle=:solid, label="")
    plot_azp = vspan!([scc_time[1], scc_time[end]], c=:black, alpha = 0.1, label = "")
    # plot_azp = plot!(xlabel="", ylabel="AZP [m]",  xlims=(0, 24), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, legendborder=:false, legend=:best, bottom_margin=2*Plots.mm, size=(600, 550))
    plot_azp = plot!(xlabel="", ylabel="AZP [m]",  xlims=(0, 96), xticks=(0:24:96), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=16, legendfont=14, legendborder=:false, legend=:none, bottom_margin=2*Plots.mm, fontfamily="Computer Modern", size=(600, 600))
    plot_scc = plot()
    plot_scc = plot!(collect(1:nt), f_scc, c=:red3, seriestype=:line, linewidth=2, linestyle=:solid, label="")
    plot_scc = vspan!([scc_time[1], scc_time[end]], c=:black, alpha = 0.1, label = "")
    # plot_scc = plot!(xlabel="Time step", ylabel=L"SCC $[\%]$", xlims=(0, 24), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, legend=:none, size=(600, 550))
    plot_scc = plot!(xlabel="Time step", ylabel="SCC [%]", xlims=(0, 96), xticks=(0:24:96), xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16, legendfont=14, legend=:none, fontfamily="Computer Modern", size=(600, 600))
    plot(plot_azp, plot_scc, layout=(2, 1))

    # ylims=(30, 55)
    # ylims=(0, 50)
    # xticks=(0:24:96)
    # size=(425, 500)
end



### pressure variation plot (cdf) ###
begin
    pv = zeros(nn)
    if pv_type == "variation"
        pv = [maximum([abs(hk[i, k] - hk[i, k-1]) for k ∈ collect(2:nt)]) for i ∈ collect(1:nn)]
    elseif pv_type == "variability"
        pv = [sqrt(sum((hk[i, k] - hk[i, k-1])^2 for k ∈ collect(2:nt))) for i ∈ collect(1:nn)]
    elseif pv_type == "range"
        pv = [maximum(hk[i, :]) - minimum(hk[i, :]) for i ∈ collect(1:nn)]
    end

    pv_cdf = sort(vec(pv))
    x = collect(1:nn)./(nn)

    # plotting code
    plot_cdf = plot()
    plot_cdf = plot!(pv_cdf, x, seriestype=:line, c=:red3, linewidth=2, linestyle=:solid, label="")
    plot_cdf = plot!(xlabel="Nodal pressure variation [m]", ylabel="Cumulative probability", yticks=(0:0.2:1), xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16, legendfont=14, legend=:none, fontfamily="Computer Modern", size=(550, 400))
    if pv_type == "variation"
        plot_cdf = plot!(xlabel="Maximum pressure variation [m]")
    elseif pv_type == "variability"
        plot_cdf = plot!(xlabel="Pressure variability [m]")
    elseif pv_type == "range"
        plot_cdf = plot!(xlabel="Pressure range [m]")
    end
    # size=(400, 300)
end