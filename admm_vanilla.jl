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
    # net_name = "bwfl_2022_05_hw"
    # net_name = "L_town"
    net_name = "modena"

    n_v = 3
    n_f = 4

    pv_type = "range" # pv_type = "variation"; pv_type = "variability"; pv_type = "range"; pv_type = "none"
    δmax = 15

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
    x_0 = SharedArray(vcat(data["q_init"], data["h_init"], zeros(np, nt), zeros(nn, nt)))
    x_k = SharedArray(zeros(np+nn+np+nn, nt))
    h̄_k   = SharedArray(data["h_init"])
    y_k = SharedArray(zeros(data["nn"], data["nt"]))
    @everywhere γ_k = 0.001 # regularisation term
    @everywhere γ_0 = 0 # regularisation term for first admm iteration
    @everywhere scaled = false # scaled = true

    # ADMM parameters
    kmax = 5000
    dim_couple = nn * nt
    ϵ_p = 1e-2
    ϵ_d = 1e-5
    obj_hist = SharedArray(zeros(kmax, nt))
    h̄_hist = Array{Union{Nothing, Float64}}(nothing, nn*nt, kmax+1)
    h̄_hist[:, 1] = vec(h̄_k)
    x_hist = Array{Union{Nothing, Float64}}(nothing, (2*np+2*nn)*nt, kmax+1)
    x_hist[:, 1] = vec(x_0)
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
                @everywhere γ = γ_0
            else
                @everywhere γ = γ_k
            end
            @sync @distributed for t ∈ collect(1:nt)
            # for t ∈ collect(1:nt)
                x_k[:, t], obj_hist[k, t], status = primal_update(x_0[:, t], h̄_k[:, t], y_k[:, t], data, γ, t, scc_time; ρ=ρ, umin=umin, δmax=δmax, scaled=scaled)
                if status != 0
                    resto = true
                    x_k[:, t], obj_hist[k, t], status = primal_update(x_0[:, t], h̄_k[:, t], y_k[:, t], data, γ, t, scc_time; ρ=ρ, umin=umin, δmax=δmax, resto=resto, scaled=scaled)
                    if status != 0
                        error("IPOPT did not converge at time step t = $t.")
                    end
                end
            end

            ### save xk data ###
            x_0 = x_k
            x_hist[:, k+1] = vec(x_k)

            ### update auxiliary variable zk ###
            h̄_k = auxiliary_update(x_0, h̄_k, y_k, data, γ_k, pv_type; δmax=δmax, scaled=scaled)
            h̄_hist[:, k+1] = vec(h̄_k)

            ### update dual variable λk ###
            h_k = x_0[np+1:np+nn, :]
            y_k = y_k + γ_k.*(h_k .- h̄_k)
            # y_k[findall(x->x .< 0, y_k)] .= 0

            ### compute residuals ### 
            p_residual_k = norm(h_k .- h̄_k) ./ sqrt(dim_couple)
            push!(p_residual, p_residual_k)
            d_residual_k = norm(γ_k * (h̄_hist[:, k+1] .- h̄_hist[:, k])) ./ sqrt(dim_couple)
            push!(d_residual, d_residual_k)

            ### ADMM status statement ###
            if p_residual[k] ≤ ϵ_p
            # if p_residual[k] ≤ ϵ_p || d_residual[k] ≤ ϵ_d
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
    x_0 = reshape(x_hist[:, 2], 2*np+2*nn, nt)
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
        q_0 = x_0[1:np, :]
        q_k = x_k[1:np, :]
        h_0 = x_0[np+1:np+nn, :]
        h_k = x_k[np+1:np+nn, :]
        A = 1 ./ ((π/4).*data["D"].^2)
        f_val = zeros(nt)
        f_azp = zeros(nt)
        f_azp_pv = zeros(nt)
        f_scc = zeros(nt)
        f_scc_pv = zeros(nt)
        for k ∈ 1:nt
            f_azp[k] = sum(data["azp_weights"][i]*(h_0[i, k] - data["elev"][i]) for i ∈ 1:nn)
            f_azp_pv[k] = sum(data["azp_weights"][i]*(h_k[i, k] - data["elev"][i]) for i ∈ 1:nn)
            f_scc[k] = sum(data["scc_weights"][j]*((1+exp(-ρ*((q_0[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ*(-(q_0[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
            f_scc_pv[k] = sum(data["scc_weights"][j]*((1+exp(-ρ*((q_k[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ*(-(q_k[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
            if k ∈ scc_time
                f_val[k] = f_scc_pv[k]*-1
            else
                f_val[k] = f_azp_pv[k]
            end
        end
    end
end


r_k = maximum(h_k, dims=2) - minimum(h_k, dims=2)
r_0 = maximum(h_0, dims=2) - minimum(h_0, dims=2)

max_viol = maximum(r_k) - δmax

histogram(r_k, bins=50)
histogram(r_0, bins=50)

sum(f_val)


iter_f
### save data ###
begin
    @save "data/admm_results/"*net_name*"_"*pv_type*"_delta_"*string(δmax)*"_gamma_"*string(γ_k)*"_distributed.jld2" nt np nn x_k x_0 objk p_residual d_residual cpu_time f_azp f_azp_pv f_scc f_scc_pv f_val iter_f max_viol
end

### load data ###
begin
    @load "data/admm_results/"*net_name*"_"*pv_type*"_delta_"*string(δmax)*"_gamma_"*string(γ_k)*"_distributed.jld2"  nt np nn x_k x_0 objk p_residual d_residual cpu_time f_azp f_azp_pv f_scc f_scc_pv f_val iter_f max_viol
end
