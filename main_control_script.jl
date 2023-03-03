#################### LOAD PACKAGES ####################

using Revise
using OpWater
using JuMP
using Ipopt
using Gurobi
using LinearAlgebra
using FileIO
using Plots; pyplot()
using LaTeXStrings
using Statistics
using JLD2


# include files with functions
include("src/vc_functions.jl")
include("src/admm_functions.jl")



#################### PRELIMINARIES ####################

### load problem data ###
begin
    net_name = "bwfl_2022_05/hw"
    # net_name = "modena"

    if net_name == "bwfl_2022_05/hw"
        data_name = "bwfl_2022_05_hw"
    end

    # load network properties
    network = load_network(net_name, afv_idx=false, dbv_idx=false, pcv_idx=false, bv_open=true)
end

### setup optimization parameters ###
begin
    n_v = 3
    n_f = 4
    αmax = 25
    δmax = 10
    umin = 0.2
    ρ = 50
    scc_time = collect(38:42) # bwfl (peak)
    # scc_time = collect(12:16) # bwfl (min)
    # scc_time = collect(7:8) # modena (peak)
    # scc_time = collect(3:4) # modena (min)
    obj_type = "azp-scc"
    type = "peak"
    opt_params = make_prob_data(network; αmax_mul=αmax, umin=umin, ρ=ρ)
end

### update variable bounds ###
begin
    q_init, h_init, err, iter = hydraulic_simulation(network, opt_params)
    S = (π * (network.D) .^ 2) / 4
    v0 = q_init ./ (1000 * S) 
    max_v = ceil(maximum(abs.(v0)))
    opt_params.Qmin , opt_params.Qmax, opt_params.umax = q_bounds_from_u(network, q_init; max_v=max_v)
end

### load actuator locations from single-objective problem ###
begin
    sol_best_azp = load("data/single_objective_results/"*data_name*"_azp_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2", "sol_best")
    sol_best_scc = load("data/single_objective_results/"*data_name*"_scc_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2", "sol_best")
    z_loc = sol_best_azp.z
    y_loc = sol_best_scc.y
end

### reassign control valve bounds ### 
begin
    # pcv actuators
    if n_v > 0
        opt_params.ηmin[setdiff(1:network.np, z_loc), :] .= 0
        opt_params.ηmax[setdiff(1:network.np, z_loc), :] .= 0
    else
        opt_params.ηmin .= 0
        opt_params.ηmax .= 0
    end

    # afv actuators
    if n_f > 0 
        opt_params.αmax[setdiff(1:network.nn, y_loc), :] .= 0
        opt_params.αmax[:, setdiff(1:network.nt, scc_time)] .= 0
    else
        opt_params.αmax .= 0
    end
end



#################### ADMM ALGORITHM ####################

### define ADMM parameters and starting values ###
# - primal variable, x := [q, h, η, α]
# - auxiliary (coupling) variable, z := h
# - dual variable λ
# - regularisation parameter γ
# - convergence tolerance ϵ

begin
    np, nn = size(network.A12)
    nt = network.nt

    # initialise variables
    xk = vcat(q_init, h_init, zeros(np, nt), zeros(nn, nt))
    zk = h_init
    λk = zeros(nn, nt)
    γk = 2 # regularisation term
    γ0 = 0 # regularisation term for first admm iteration

    # ADMM parameters
    kmax = 100
    ϵ = 2e-1
    obj_hist = Array{Union{Nothing, Float64}}(nothing, kmax, nt)
    z_hist = Array{Union{Nothing, Float64}}(nothing, nn*nt, kmax+1)
    z_hist[:, 1] = vec(zk)
    x_hist = Array{Union{Nothing, Float64}}(nothing, (2*np+2*nn)*nt, kmax+1)
    x_hist[:, 1] = vec(xk)
    p_residual = []
    d_residual = []

end

### main ADMM loop ###
begin
    cpu_time = @elapsed begin
        for k ∈ collect(1:kmax)

            ### update (in parallel) primal variable xk_t ###
            # set regularisation parameter γ
            if k == 1
                γ = γ0
            else
                γ = γk
            end
            Threads.@threads for t ∈ collect(1:nt)
            # for t ∈ collect(1:nt)
                # @show "" t Threads.threadid()
                xk[:, t], obj_hist[k, t], status = primal_update(xk[:, t], zk[:, t], λk[:, t], network, opt_params, γ, t, scc_time, z_loc, y_loc; ρ=ρ, umin=umin, δmax = δmax)
                if status != 0 
                    error("IPOPT did not converge at time step t = $t.")
                end
            end

            ### save xk data ###
            x_hist[:, k+1] = vec(xk)

            ### update auxiliary variable zk ###
            zk = auxiliary_update(xk, zk, λk, network, opt_params, γk; δmax=δmax)
            z_hist[:, k+1] = vec(zk)

            ### update dual variable λk ###
            hk = xk[np+1:np+nn, :]
            λk = λk + γk.*(hk - zk)
            # λk[findall(x->x .< 0, λk)] .= 0

            ### compute residuals ### 
            p_residual_k = maximum(abs.(hk - zk))
            push!(p_residual, p_residual_k)
            d_residual_k = maximum(abs.(z_hist[:, k+1] - z_hist[:, k]))
            push!(d_residual, d_residual_k)

            ### ADMM status statement ###
            if p_residual[k] ≤ ϵ && d_residual[k] ≤ ϵ
                break
                @info "ADMM successful at iteration $k of $kmax. Algorithm terminated."
            else
                @info "ADMM unsuccessful at iteration $k of $kmax. Moving to next iteration."
            end

        end
    end
end


### save data ###
begin
    @save "data/adaptive_control_results/"*data_name*"_adaptive_control_admm_nv_"*string(n_v)*"_nf_"*string(n_f)*"_pvmax_"*string(δmax)*".jld2" xk x_hist p_residual d_residual cpu_time
end

### load data ###
begin
    @load "data/adaptive_control_results/"*data_name*"_adaptive_control_admm_nv_"*string(n_v)*"_nf_"*string(n_f)*"_pvmax_"*string(δmax)*".jld2" xk x_hist p_residual d_residual cpu_time
end

### plotting code ###
begin
    PyPlot.rc("text", usetex=true)
    PyPlot.rc("font", family="CMU Serif")

    ### plot residuals ###
    plot_p_residual = plot()
    plot_p_residual = plot!(collect(1:length(p_residual)), p_residual, c=:blue, markerstrokecolor=:blue, seriestype=:scatter, markersize=4)
    plot_p_residual = plot!(xlabel="", ylabel="Primal residual [m]", ylims=(0, 10), xlims=(0, 25), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, legend=:none, bottom_margin=2*Plots.mm, size = (425, 500))
    plot_d_residual = plot()
    plot_d_residual = plot!(collect(1:length(d_residual)), d_residual, c=:blue, markerstrokecolor=:blue, seriestype=:scatter, markersize=4)
    plot_d_residual = plot!(xlabel="ADMM iteration", ylabel="Dual residual [m]", ylims=(0, 50), xlims=(0, 25), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, legend=:none, size = (425, 500))
    plot(plot_p_residual, plot_d_residual, layout = (2, 1))
end

### plot objective function (time series) ### 
begin
    xk_0 = reshape(x_hist[:, 2], 2*np+2*nn, nt)
    qk_0 = xk_0[1:np, :]
    qk = xk[1:np, :]
    hk_0 = xk_0[np+1:np+nn, :]
    hk = xk[np+1:np+nn, :]
    A = 1 ./ ((π/4).*network.D.^2)
    f_azp = zeros(nt)
    f_azp_pv = zeros(nt)
    f_scc = zeros(nt)
    f_scc_pv = zeros(nt)
    for k ∈ 1:nt
        f_azp[k] = sum(opt_params.azp_weights[i]*(hk_0[i, k] - network.elev[i]) for i ∈ 1:nn)
        f_azp_pv[k] = sum(opt_params.azp_weights[i]*(hk[i, k] - network.elev[i]) for i ∈ 1:nn)
        f_scc[k] = sum(opt_params.scc_weights[j]*((1+exp(-ρ*((qk_0[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ*(-(qk_0[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
        f_scc_pv[k] = sum(opt_params.scc_weights[j]*((1+exp(-ρ*((qk[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ*(-(qk[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
    end

    plot_azp = plot()
    plot_azp = plot!(collect(1:nt), f_azp, c=:blue, seriestype=:line, linewidth=1.5, linestyle=:dash, label="without PV")
    plot_azp = plot!(collect(1:nt), f_azp_pv, c=:blue, seriestype=:line, linewidth=1.5, label="with PV")
    plot_azp = vspan!([38, 42]; c=:black, alpha = 0.1, label = "SCC period")
    plot_azp = plot!(xlabel="", ylabel="AZP [m]", ylims=(30, 55), xlims=(0, 96), xticks=(0:24:96), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, legendborder=:false, legend=:best, bottom_margin=2*Plots.mm, size = (425, 500))
    plot_scc = plot()
    plot_scc = plot!(collect(1:nt), f_scc, c=:blue, seriestype=:line, linewidth=1.5, linestyle=:dash, label="without PV")
    plot_scc = plot!(collect(1:nt), f_scc_pv, c=:blue, seriestype=:line, linewidth=1.5, label="with PV")
    plot_scc = vspan!([38, 42]; c=:black, alpha = 0.1, label = "SCC period")
    plot_scc = plot!(xlabel="Time step", ylabel=L"SCC $[\%]$", ylims=(0, 50),xlims=(0, 96), xticks=(0:24:96), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, legend=:none, size=(425, 500))
    plot(plot_azp, plot_scc, layout=(2, 1))
end


### nodal pressure variation (time series) ###
begin
    pv_0 = zeros(nn, nt)
    pv_k = zeros(nn, nt)
    pv_0[:, 1] .= NaN
    pv_k[:, 1] .= NaN
    for i ∈ collect(2:nt)
        pv_0[:, i] = hk_0[:, i] - hk_0[:, i-1]
        pv_k[:, i] = hk[:, i] - hk[:, i-1]
    end

    pv_0_mean = vec(mean(pv_0, dims=1))
    pv_0_min = vec(minimum(pv_0, dims=1))
    pv_0_max = vec(maximum(pv_0, dims=1))

    pv_k_mean = vec(mean(pv_k, dims=1))
    pv_k_min = vec(minimum(pv_k, dims=1))
    pv_k_max = vec(maximum(pv_k, dims=1))

    # plotting code
    plot_npv_0 = plot()
    plot_npv_0 = plot!(collect(1:nt), pv_0_mean, c=:blue, linestyle=:dash, linewidth=1.5, label="without PV")
    plot_npv_0 = plot!(collect(1:nt), pv_0_min, fillrange = pv_0_max, fillalpha = 0.15, c=:blue, linealpha = 0, label="")
    plot_npv_0 = plot!(xlabel="", ylabel="Nodal PV [m]", ylims=(-20, 30), xlims=(0, 96), xticks=(0:24:96), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, size=(425, 500), legend=:topright)
    plot_npv_k = plot()
    plot_np_k = plot!(collect(1:nt), pv_k_mean, c=:blue, linestyle=:solid, linewidth=1.5, label="with PV")
    plot_npv_k = plot!(collect(1:nt), pv_k_min, fillrange = pv_k_max, fillalpha = 0.15, c=:blue, linealpha = 0, label="")
    plot_npv_k = plot!(xlabel="Time step", ylabel="Nodal PV [m]", ylims=(-20, 30), xlims=(0, 96), xticks=(0:24:96), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, size=(425, 500), legend=:topright)
    plot(plot_npv_0, plot_npv_k, layout=(2, 1))

end

### nodal pressure variation (cdf) ###
begin
    pv_0 = zeros(nn, nt-1)
    pv_k = zeros(nn, nt-1)
    for i ∈ collect(1:nt-1)
        pv_0[:, i] = hk_0[:, i+1] - hk_0[:, i]
        pv_k[:, i] = hk[:, i+1] - hk[:, i]
    end

    pv_0_cdf = sort(vec(pv_0))
    pv_k_cdf = sort(vec(pv_k))
    x = collect(1:nn*nt)./(nn.*nt)

    # plotting code
    plot_cdf = plot()
    plot_cdf = plot!(pv_k_cdf, x, seriestype=:line, c=:blue, linewidth=1.5, linestyle=:solid, label="with PV")
    plot_cdf = plot!(pv_0_cdf, x, seriestype=:line, c=:blue, linewidth=1.5, linestyle=:dot, label="without PV")
    plot_cdf = plot!(xlabel="Nodal pressure variation [m]", ylabel="Cumulative distribution", yticks=(0:0.2:1), xtickfontsize=14, ytickfontsize=14, xguidefontsize=14, yguidefontsize=14, legendfont=12, size=(400, 300), legend=:false)

end

### nodal pressure variation (network) ###
begin
    npv_0 = zeros(nn)
    npv_k = zeros(nn)
    for i ∈ collect(1:nn)
        npv_0[i] = maximum(hk_0[i, :]) - minimum(hk_0[i, :])
        npv_k[i] = maximum(hk[i, :]) - minimum(hk[i, :])
    end
    network.pcv_loc = z_loc
    network.afv_loc = y_loc
    node_key = "pressure variation"
    node_values = vcat(npv_k, repeat([0.0], network.n0))
  
    # plot_network_nodes(network, node_values=node_values, node_key=node_key)
    plot_network_nodes(network, node_values=node_values, node_key=node_key)
    # plot_network_edges(network, edge_values=edge_values, edge_key=edge_key, clims=(0, 0.4))
    plot_network_layout(network, pipes=false, reservoirs=true, pcvs=true, dbvs=false, afvs=true)

end