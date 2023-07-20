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
using FileIO
using JLD2

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
    include("src/two_level_functions.jl")
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
    δmax = 100

end


### load problem data for distributed.jl version ###
@everywhere begin
    data = load("data/problem_data/"*net_name*"_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2")
end



#################### TWO-LEVEL DISTRIBUTED ALGORITHM ####################

### define ADMM parameters and starting values ###
# - primal variables, x := [q, h, η, α]
# - auxiliary (coupling) variable, h̄  := h
# - slack variable, z
# - dual variable λ for slack variable constraint z = 0
# - β > 0 ALM penalty parameter
# - dual variable y for concensus constraint Ah + Bh̄ + z = 0. where A = identity matrix and B = - identity matrix
# - γ > 0 ADMM penalty parameter
# - convergence tolerance, ϵ


### initialise algorithm ###
begin
    
    # unload data
    np = data["np"]
    nn = data["nn"]
    nt = data["nt"]
    scc_time = data["scc_time"]
    ρ_scc = data["ρ"]
    umin = data["umin"]

    # initialise outer ALM level variables
    λ_n = SharedArray(zeros(data["nn"], data["nt"]))
    @everywhere β_n = 0.1
    β_0 = β_n

    # initialise inner ADMM level variables
    x_0 = SharedArray(vcat(data["q_init"], data["h_init"], zeros(np, nt), zeros(nn, nt)))
    x_k = SharedArray(zeros(np+nn+np+nn, nt))
    h̄_k = SharedArray(data["h_init"])
    z_k = SharedArray(zeros(data["nn"], data["nt"]))
    y_k = SharedArray(zeros(data["nn"], data["nt"]))
    @everywhere ρ_k = 0

    # algorithm parameters
    max_iter = 1000 # inner layer iterations
    n_iter = 1
    k_iter = 1
    λ_bound = 1e6
    dim_couple = nn * nt
    ϵ_p = 1e-2
    ϵ_d = 5e-3
    @everywhere γ = 1.1
    ω = 0.8
    res_z_prev = 0

    # initialise data arrays
    obj_hist = SharedArray(zeros(max_iter, nt))
    x_hist = Array{Union{Nothing, Float64}}(nothing, (2*np+2*nn)*nt, max_iter); x_hist[:, k_iter] = vec(x_0)
    h̄_hist = Array{Union{Nothing, Float64}}(nothing, nn*nt, max_iter); h̄_hist[:, k_iter] = vec(h̄_k)
    z_hist = Array{Union{Nothing, Float64}}(nothing, nn*nt, max_iter); z_hist[:, k_iter] = vec(z_k)
    y_hist = Array{Union{Nothing, Float64}}(nothing, nn*nt, max_iter); y_hist[:, k_iter] = vec(y_k)
    ρ_hist = Array{Union{Nothing, Float64}}(nothing, 1, max_iter); ρ_hist[:, k_iter] .= ρ_k
    λ_hist = Array{Union{Nothing, Float64}}(nothing, nn*nt, max_iter); λ_hist[:, n_iter] = vec(λ_n)
    β_hist = Array{Union{Nothing, Float64}}(nothing, 1, max_iter); β_hist[:, n_iter] .= β_n

    residuals = Array{Union{Nothing, Float64}}(nothing, max_iter, 5); residuals[1, :] = zeros(1, 5) # residuals corresponding to equations (14a)--(14c) in Sun, K and Sun, X. (2023)

end


### implement algorithm ###
cpu_time = @elapsed begin
    
    while k_iter ≤ max_iter

        ## inner ADMM level updates ##

        # Step 1: update x block in parallel
        @sync @distributed for t ∈ collect(1:nt)
            x_k[:, t], obj_hist[k_iter+1, t], status = x_update(x_0[:, t], h̄_k[:, t], z_k[:, t], y_k[:, t], λ_n[:, t], data, β_n, ρ_k, t, scc_time; ρ_scc=ρ_scc, umin=umin, δmax=δmax)
            if status != 0
                resto = true
                x_k[:, t], obj_hist[k_iter, t], status = x_update(x_0[:, t], h̄_k[:, t], z_k[:, t], y_k[:, t], λ_n[:, t], data, β_n, ρ_k, t, scc_time; ρ_scc=ρ_scc, umin=umin, δmax=δmax, resto=resto)
                if status != 0
                    error("IPOPT did not converge at time step t = $t.")
                end
            end
        end

        x_hist[:, k_iter+1] = vec(x_k) # save x data at current k iteration
        if k_iter == 1
            @everywhere ρ_k = 2 * β_n
        end

        # Step 2: update h̄ block (note that this couples time steps and most be solved centrally)
        h̄_k = h̄_update(x_k, h̄_k, z_k, y_k, λ_n, data, β_n, ρ_k, pv_type; δmax=δmax)
        h̄_hist[:, k_iter+1] = vec(h̄_k)

        # Step 3: update z block (note that this is an unconstrained optimization problem)
        z_k = z_update(x_k, h̄_k, z_k, y_k, λ_n, data, β_n, ρ_k)
        z_hist[:, k_iter+1] = vec(z_k)

        # Step 4: update inner level dual variable y_k
        h_k = x_k[np+1:np+nn, :]
        y_k = y_k .+ ρ_k .* (h_k .- h̄_k .+ z_k)
        y_hist[:, k_iter+1] = vec(y_k)

        # Step 5: compute residuals
        res_inner_a = norm(h̄_hist[:, k_iter+1] .- h̄_hist[:, k_iter] .+ z_hist[:, k_iter] .- z_hist[:, k_iter+1]) ./ sqrt(dim_couple) # (14a)
        res_inner_b = norm(z_hist[:, k_iter+1] .- z_hist[:, k_iter]) ./ sqrt(dim_couple) # (14b)
        res_inner_c = norm(h_k .- h̄_k .+ z_k) ./ sqrt(dim_couple) # (14c)
        res_outer = norm(h_k .- h̄_k) ./ sqrt(dim_couple) # ||h_k - h̄_k|| 
        res_z = norm(z_k) # ||z_k|| 

        residuals[k_iter+1, :] = hcat(res_inner_a, res_inner_b, res_inner_c, res_outer, res_z)

        # Step 6: check stopping criteria
        # if (res_inner_a ≤ ϵ) && (res_inner_b ≤ ϵ) && (res_inner_c ≤ 2 * ϵ)
        if (res_inner_c ≤ 1 ./ (2500 .* n_iter)) || ((res_inner_b ≤ ϵ_d) && k_iter > 1)
            @info "ADMM successful at iteration $k_iter. Inner-level residual (a) = $res_inner_a. Inner-level residual (b) = $res_inner_b, Inner-level residual (c) = $res_inner_c. ALM iteration $n_iter finished."

            # check overall algorithm stopping criterion
            if res_outer ≤ ϵ_p
                @info "Algorithm successfuly converged. Outer-level residual = $res_outer. ALM iteration = $n_iter."
                break
            else
                @info "Algorithm unsuccessful at ALM iteration $n_iter. Outer-level residual = $res_outer. Moving to next iteration."
            end

            # Step 7: update outer dual variable λ_k
            λ_n = λ_update(λ_n, z_k, β_n, λ_bound)
            λ_hist[:, n_iter+1] = vec(λ_n)

            # Step 8: update penalty terms β_k and ρ_k
            if (res_z > ω * res_z_prev) && (β_n * γ ≤ 1e6) && (n_iter > 1)
                @everywhere β_n = β_n * γ
                @everywhere ρ_k = 2 * β_n
            end

            n_iter += 1
            res_z_prev = res_z

            # Step 9: initialise ADMM level
            z_k = SharedArray(zeros(data["nn"], data["nt"]))
            y_k = -1 .* λ_n
            x_0 = x_k
            h̄_k = h_k

        else
            @info "ADMM unsuccessful at iteration $k_iter. Inner-level residual (a) = $res_inner_a. Inner-level residual (b) = $res_inner_b, Inner-level residual (c) = $res_inner_c. Moving to next iteration."
            x_0 = x_k

        end

        k_iter += 1

    end

end


### print algorithm values ###
begin
    println("")
    println("Number of inner ADMM iterations: $k_iter")
    println("Number of outer ALM  iterations: $n_iter")
    println("ALM penalty parameter at termination: $β_n")
    println("The two-level distributed algorithm finished in $cpu_time seconds.")
end

### compute objective function (time series) ### 
begin
    if k_iter == max_iter
        f_val = Inf
        f_azp = Inf
        f_azp_pv = Inf
        f_scc = Inf
        f_scc_pv = Inf
        cpu_time = Inf
    else
        x_0 = reshape(x_hist[:, 1], 2*np + 2*nn, nt)
        x_k = reshape(x_hist[:, k_iter+1], 2*np + 2*nn, nt)
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
            f_scc[k] = sum(data["scc_weights"][j]*((1+exp(-ρ_scc*((q_0[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ_scc*(-(q_0[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
            f_scc_pv[k] = sum(data["scc_weights"][j]*((1+exp(-ρ_scc*((q_k[j, k]/1000*A[j]) - umin)))^-1 + (1+exp(-ρ_scc*(-(q_k[j, k]/1000*A[j]) - umin)))^-1) for j ∈ 1:np)
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


### save data ###
begin
    @save "data/two_level_results/"*net_name*"_"*pv_type*"_"*string(δmax)*"_beta_"*string(β_0)*".jld2" nt np nn x_k x_0 obj_hist residuals k_iter n_iter cpu_time f_azp f_azp_pv f_scc f_scc_pv f_val max_viol
end

### load data ###
begin
    @load "data/two_level_results/"*net_name*"_"*pv_type*"_"*string(δmax)*"_beta_"*string(β_0)*".jld2" nt np nn x_k x_0 obj_hist residuals k_iter n_iter cpu_time f_azp f_azp_pv f_scc f_scc_pv f_val max_viol
end