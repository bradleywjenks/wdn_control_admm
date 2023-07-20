#################### LOAD DEPENDENCIES ####################

# restart procs
rmprocs(workers())

### load dependencies for main worker
using Distributed
using SharedArrays
using LaTeXStrings
using Statistics
using Plots
using LinearAlgebra

# OPENBLAS_NUM_THREADS = 1
addprocs(7)

### instantiate and precompile environment
@everywhere begin
    using Pkg; Pkg.activate(".")
    Pkg.instantiate(); Pkg.precompile()
end

### load dependencies for local workers
@everywhere begin
    using FileIO
    using JLD2
    include("src/admm_functions.jl")
end



#################### PRELIMINARIES ####################

### input problem parameters ###
@everywhere begin
    net_name = "bwfl_2022_05_hw"
    # net_name = "L_town"
    # net_name = "modena"

    n_v = 3
    n_f = 4

    pv_type = "range" # pv_type = "variation"; pv_type = "variability"; pv_type = "range"; pv_type = "none"
    δmax = 10

end



### load problem data for distributed.jl version ###
@everywhere begin
    data = load("data/problem_data/"*net_name*"_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2")
end



#################### ADMM ALGORITHM ####################

### define ADMM parameters and starting values ###
# - primal variable, x := [q, h, η, α]
# - auxiliary (coupling) variable, z := h
# - dual variable λ
# - penalty parameter γ
# - convergence tolerance ϵ

begin

    # unload data
    np = data["np"]
    nn = data["nn"]
    nt = data["nt"]
    scc_time = data["scc_time"]
    ρ = data["ρ"]
    umin = data["umin"]

    # initialise variables
    xk_0 = SharedArray(vcat(data["q_init"], data["h_init"], zeros(np, nt), zeros(nn, nt)))
    zk = SharedArray(data["h_init"])
    λk = SharedArray(zeros(data["nn"], data["nt"]))
    @everywhere γk = 1 # regularisation term
    @everywhere γ0 = 0 # regularisation term for first admm iteration
    @everywhere scaled = true # scaled = false

    # ADMM parameters
    kmax = 1000
    ϵ_rel = 5e-3
    ϵ_abs = 1e-2
    obj_hist = SharedArray(zeros(kmax, nt))
    xk = SharedArray(zeros(np+nn+np+nn, nt))
    z_hist = Array{Union{Nothing, Float64}}(nothing, nn*nt, kmax+1)
    z_hist[:, 1] = vec(zk)
    x_hist = Array{Union{Nothing, Float64}}(nothing, (2*np+2*nn)*nt, kmax+1)
    x_hist[:, 1] = vec(xk_0)
    p_residual = []
    d_residual = []
    iter_f = []

end

### main ADMM loop ###
begin
    cpu_time = @elapsed begin
        for k ∈ collect(1:kmax)

            ### update (in parallel) primal variable xk_t ###

            # set regularisation parameter γ
            if k == 1
                @everywhere γ = γ0
            else
                @everywhere γ = γk
            end
            @sync @distributed for t ∈ collect(1:nt)
            # for t ∈ collect(1:nt)
                xk[:, t], obj_hist[k, t], status = primal_update(xk_0[:, t], zk[:, t], λk[:, t], data, γ, t, scc_time; ρ=ρ, umin=umin, δmax=δmax, scaled=scaled)
                if status != 0
                    resto = true
                    xk[:, t], obj_hist[k, t], status = primal_update(xk_0[:, t], zk[:, t], λk[:, t], data, γ, t, scc_time; ρ=ρ, umin=umin, δmax=δmax, resto=resto, scaled=scaled)
                    if status != 0
                        error("IPOPT did not converge at time step t = $t.")
                    end
                end
            end

            ### save xk data ###
            xk_0 = xk
            x_hist[:, k+1] = vec(xk)

            ### update auxiliary variable zk ###
            zk = auxiliary_update(xk_0, zk, λk, data, γk, pv_type; δmax=δmax, scaled=scaled)
            z_hist[:, k+1] = vec(zk)

            ### update dual variable λk ###
            hk = xk_0[np+1:np+nn, :]
            λk = λk + γk.*(hk .- zk)
            # λk[findall(x->x .< 0, λk)] .= 0

            ### compute residuals ### 
            p_residual_k = norm(hk .- zk)
            push!(p_residual, p_residual_k)
            d_residual_k = norm(z_hist[:, k+1] .- z_hist[:, k])
            push!(d_residual, d_residual_k)

            ### ADMM status statement ###
            # ϵ_p = sqrt(length(hk))*ϵ_abs 
            # ϵ_d = sqrt(length(λk))*ϵ_abs
            ϵ_p = sqrt(length(hk))*ϵ_abs + ϵ_rel*maximum([norm(hk), norm(zk)])
            ϵ_d = sqrt(length(λk))*ϵ_abs + ϵ_rel*norm(λk)
            if p_residual[k] ≤ ϵ_p && d_residual[k] ≤ ϵ_d
                iter_f = k
                @info "ADMM successful at iteration $k of $kmax. Primal residual = $p_residual_k, Dual residual = $d_residual_k. Algorithm terminated."
                break
            else
                iter_f = k
                @info "ADMM unsuccessful at iteration $k of $kmax. Primal residual = $p_residual_k, Dual residual = $d_residual_k. Moving to next iteration."
            end

        end
    end

    objk = obj_hist[iter_f, :]
    xk_0 = reshape(x_hist[:, 2], 2*np+2*nn, nt)
end



### compute objective function (time series) ### 
begin
    if iter_f == kmax
        f_val = Inf
        f_azp = Inf
        f_azp_pv = Inf
        f_scc = Inf
        f_scc_pv = Inf
        cpu_time = Inf
    else
        qk_0 = xk_0[1:np, :]
        qk = xk[1:np, :]
        hk_0 = xk_0[np+1:np+nn, :]
        hk = xk[np+1:np+nn, :]
        A = 1 ./ ((π/4).*data["D"].^2)
        f_val = zeros(nt)
        f_azp = zeros(nt)
        f_azp_pv = zeros(nt)
        f_scc = zeros(nt)
        f_scc_pv = zeros(nt)
        for k ∈ 1:nt
            f_azp[k] = sum(data["azp_weights"][i]*(hk_0[i, k] - data["elev"][i]) for i ∈ 1:nn)
            f_azp_pv[k] = sum(data["azp_weights"][i]*(hk[i, k] - data["elev"][i]) for i ∈ 1:nn)
            f_scc[k] = sum(data["scc_weights"][j]*((1+exp(-ρ*((qk_0[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ*(-(qk_0[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
            f_scc_pv[k] = sum(data["scc_weights"][j]*((1+exp(-ρ*((qk[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ*(-(qk[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
            if k ∈ scc_time
                f_val[k] = f_scc_pv[k]*-1
            else
                f_val[k] = f_azp_pv[k]
            end
        end
    end
end


### save data ###
begin
    @save "data/admm_results/"*net_name*"_"*pv_type*"_delta_"*string(δmax)*"_gamma_"*string(γk)*"_distributed.jld2" nt np nn xk xk_0 objk p_residual d_residual cpu_time f_azp f_azp_pv f_scc f_scc_pv f_val 
end

### load data ###
begin
    @load "data/admm_results/"*net_name*"_"*pv_type*"_delta_"*string(δmax)*"_gamma_"*string(γk)*"_distributed.jld2"  nt np nn xk xk_0 objk p_residual d_residual cpu_time f_azp f_azp_pv f_scc f_scc_pv f_val 
end
