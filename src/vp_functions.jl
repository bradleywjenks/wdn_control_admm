### valve placement (VP) functions
using FLoops
using LinearAlgebra
using SparseArrays
using StatsBase
using Parameters
using ProgressMeter

@with_kw mutable struct Solution
    q::Matrix{Union{Nothing, Float64}}
    h::Matrix{Union{Nothing, Float64}}
    η::Matrix{Union{Nothing, Float64}}
    α::Matrix{Union{Nothing, Float64}}
    v::Vector{Union{Nothing, Int}}
    y::Vector{Union{Nothing, Int}}
    obj_val::Float64
    cpu::Float64
end





function convex_heuristic(network, probdata, obj_type, heuristic; trials::Int64=100, nv::Int64=0, nf::Int64=0, num_OA_cuts::Int64=0, ω=nothing, f_data=nothing, nlp_solver="ipopt", network_decomp=nothing, bi_dir=false, v_thresh=5e-02, y_thresh=1e-04)
    
    # first, solve convex relaxation
    _, _, η_r, _, v_r, y_r, _, _ = solve_convex_relax(network, probdata, obj_type; nv=nv, nf=nf, ω=ω, f_data=f_data, network_decomp=network_decomp)
    η_init = η_r

    # sort nv placement data from convex relaxation
    if bi_dir
        v = v_r[1:network.np] .+ v_r[network.np+1:end]
    else
        v = v_r
    end
    v_idx = sortperm(v, rev=true)
    v_prob = v[v_idx]
    v_ignore = findall(x -> x ≤ v_thresh, v_prob)
    v_fix = findall(x -> x == 1, v_prob)
    deleteat!(v_prob, v_ignore)
    deleteat!(v_idx, v_ignore)
    v_count = length(v_prob) - length(v_fix)

    # sort nf placement data from convex relaxation
    y_idx = sortperm(y_r, rev=true)
    y_prob = y_r[y_idx]
    y_ignore = findall(x -> x ≤ y_thresh, y_prob)
    y_fix = findall(x -> x == 1, y_prob)
    deleteat!(y_prob, y_ignore)
    deleteat!(y_idx, y_ignore)
    y_count = length(y_prob) - length(y_fix)

    # compute number of possible v ∪ z valve configurations
    nCr = factorial(big(v_count+y_count)) / (factorial(big((v_count+y_count)-(nv+nf)))*factorial(big(nv+nf)))
    N = min(nCr, trials)
    N = convert(Int, N)

    # heuristic for fixing binary variables
    if heuristic == "random"
        x_sol, obj_sol, cpu_sol, valve_visited = random_sampling(network, probdata, obj_type, N, nv, nf, ω, f_data, nlp_solver, η_init, bi_dir, v_idx, v_prob, y_idx, y_prob)
    elseif heuristic == "swapping"
        x_sol, obj_sol, cpu_sol, valve_visited = round_and_swap(network, probdata, obj_type, N, nv, nf, ω, f_data, nlp_solver, η_init, bi_dir, v_idx, v_prob, y_idx, y_prob)
    end

    # find valve configuration corresponding to best nlp solution
    nlp_best = sortperm(filter(x -> x != nothing, obj_sol))
    nlp_best = nlp_best[1]
    # # find valve configuration corresponding to best nlp solution
    # feasibility_check = any(obj_sol == Inf)
    # if feasibility_check
    #     @error "Model infeasibility error"
    # else
    #     nlp_best = sortperm(obj_sol)
    #     nlp_best = nlp_best[1]
    # end

    # save best nlp solution data
    x_best = reshape(x_sol[:, nlp_best], network.np+network.nn+network.np+network.nn, network.nt)
    q_best = x_best[1:network.np, :]
    h_best = x_best[network.np+1:network.np+network.nn, :]
    η_best = x_best[network.np+network.nn+1:2*network.np+network.nn, :]
    α_best = x_best[2*network.np+network.nn+1:end, :]
    obj_best = obj_sol[nlp_best]
    cpu_best = cpu_sol[nlp_best]
    v_best = valve_visited[nlp_best][1:nv]
    y_best = valve_visited[nlp_best][nv+1:end]

    sol_best = Solution(q=q_best, h=h_best, η=η_best, α=α_best, v=v_best, y=y_best, obj_val=obj_best, cpu=cpu_best)

    return sol_best, obj_sol

end




function solve_convex_relax_old(network, probdata, obj_type; nv::Int64=0, nf::Int64=0, num_OA_cuts::Int64=0, ω=nothing, f_data=nothing, network_decomp=nothing)

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

    # unload problem data
    q_lo = copy(probdata.Qmin)
    q_up = copy(probdata.Qmax)
    h_lo = copy(probdata.Hmin)
    h_up = copy(probdata.Hmax)
    η_lo = copy(probdata.ηmin)
    η_up = copy(probdata.ηmax)
    α_up = copy(probdata.αmax)
    α_lo = zeros(nn, nt)

     # define HW head loss model parameters
     r = determine_r(network.L, network.D, network.C, network.nexp, valve_idx=network.valve_idx)
     ϕ_ex(q, r, nexp) = r.*q.*abs.(q).^(nexp.-1) # define head loss model

     # implement QA model later...
 
     # assign pcv and dbv valve data
     v_lo = zeros(2*np, nt)
     v_up = ones(2*np, nt)
     if !isempty(network.pcv_loc)
         v_lo[network.pcv_idx, :] .= 1
     end
 
     z_lo = zeros(np, 1)
     z_up = zeros(np, 1)
     if !isempty(network.pcv_loc)
         z_lo[network.pcv_loc] .= 1
         z_up[network.pcv_loc] .= 1
     end
     if nv > 0
         z_up = ones(np, 1)
     end
     if network_decomp !== nothing
        z_up[network_decomp.F] .= 0
     end
 
     # AFV data
     y_lo = zeros(nn, nt)
     y_up = zeros(nn, nt)
     if nf > 0
         y_up = ones(nn, nt)
     end
     
     # set θ bounds
     θ_lo = ϕ_ex(q_lo, r, nexp)
     θ_up = ϕ_ex(q_up, r, nexp)

    # build optimization model
    model = Model(Gurobi.Optimizer)
    # set_optimizer_attribute(model,"Method", 2)
    set_optimizer_attribute(model,"Presolve", 0)
    set_optimizer_attribute(model,"Crossover", 0)
    # set_optimizer_attribute(model,"NumericFocus", 3)
    set_silent(model)
    
    # define variables
    @variable(model, q_lo[i, k] ≤ q[i=1:np, k=1:nt] ≤ q_up[i, k])
    @variable(model, h_lo[i, k] ≤ h[i=1:nn, k=1:nt] ≤ h_up[i, k])
    @variable(model, η_lo[i, k] ≤ η[i=1:np, k=1:nt] ≤ η_up[i, k])
    @variable(model, θ_lo[i, k] ≤ θ[i=1:np, k=1:nt] ≤ θ_up[i, k])
    @variable(model, α_lo[i, k] ≤ α[i=1:nn, k=1:nt] ≤ α_up[i, k])
    @variable(model, v_lo[i, k] ≤ v[i=1:2*np, k=1:nt] ≤ v_up[i, k])
    @variable(model, z_lo[i] ≤ z[i=1:np] ≤ z_up[i])
    @variable(model, y_lo[i] ≤ y[i=1:nn] ≤ y_up[i])

    # hydraulic constraints
    @constraint(model, θ + A12*h + A10*h0 + η .== 0)
    @constraint(model, A12'*q - α .== d)

    # big-M valve contraints
    @constraint(model, [i=1:np, k=1:nt], η[i, k] -η_up[i, k]*v[i, k] ≤ 0)
    @constraint(model, [i=1:np, k=1:nt], -η[i, k] +η_lo[i, k]*v[np+i, k] ≤ 0)
    @constraint(model, [i=1:np, k=1:nt], -q[i, k] -q_lo[i, k]*v[i, k] ≤ -q_lo[i, k])
    @constraint(model, [i=1:np, k=1:nt], q[i, k] +q_up[i, k]*v[np+i, k] ≤ q_up[i, k])
    @constraint(model, [i=1:np, k=1:nt], -θ[i, k] -θ_lo[i, k]*v[i, k] ≤ -θ_lo[i, k])
    @constraint(model, [i=1:np, k=1:nt], θ[i, k] +θ_up[i, k]*v[np+i, k] ≤ θ_up[i, k])
    @constraint(model, [i=1:nn, k=1:nt], α[i, k] -α_up[i, k]*y[i] ≤ 0)

    # valve placement constraints
    @constraint(model, sum(z) == nv)
    @constraint(model,[i=1:np, k=1:nt], v[i, k]+v[np+i, k] == z[i])
    @constraint(model, sum(y) == nf)
    
    # head loss model relaxation constraints
    R, E, R_rhs = friction_HW_relax(q_lo, q_up, r, nexp)
    @constraint(model, ϕ_relax1[i=1:np, k=1:nt], R[i, k, 1] * q[i, k] + E[i, k, 1] * θ[i, k] ≤ R_rhs[i, k, 1] )
    @constraint(model, ϕ_relax2[i=1:np, k=1:nt], R[i, k, 2] * q[i, k] + E[i, k, 2] * θ[i, k] ≤ R_rhs[i, k, 2] )
    @constraint(model, ϕ_relax3[i=1:np, k=1:nt], R[i, k, 3] * q[i, k] + E[i, k, 3] * θ[i, k] ≤ R_rhs[i, k, 3] )
    @constraint(model, ϕ_relax4[i=1:np, k=1:nt], R[i, k, 4] * q[i, k] + E[i, k, 4] * θ[i, k] ≤ R_rhs[i, k, 4] )

    # additional model terms for SCC optimization problem
    if obj_type == "scc" || obj_type == "azp-scc"
        # sigmoid function auxiliary variables
        @variable(model, 0 ≤ σ⁺[i=1:network.np, k=1:network.nt] ≤ 1)
        @variable(model, 0 ≤ σ⁻[i=1:network.np, k=1:network.nt] ≤ 1)

        # sigmoid function relaxation constraints
        pipe_csa =  1000*(π * (network.D) .^ 2) / 4
        S, T, S_rhs = sigmoid_relax(q_lo, q_up, opt_params.umax, opt_params.umin, pipe_csa, opt_params.ρ)
        @constraint(model, ψ⁺_relax1[i=1:np, k=1:nt], S[i, k, 1] * q[i, k] + T[i, k, 1] * σ⁺[i, k] ≤ S_rhs[i, k, 1] )
        @constraint(model, ψ⁺_relax2[i=1:np, k=1:nt], S[i, k, 2] * q[i, k] + T[i, k, 2] * σ⁺[i, k] ≤ S_rhs[i, k, 2] )
        @constraint(model, ψ⁻_relax1[i=1:np, k=1:nt], S[i, k, 3] * q[i, k] + T[i, k, 3] * σ⁻[i, k] ≤ S_rhs[i, k, 3] )
        @constraint(model, ψ⁻_relax2[i=1:np, k=1:nt], S[i, k, 4] * q[i, k] + S[i, k, 4] * σ⁻[i, k] ≤ S_rhs[i, k, 4] )
    end
    
    # assign model objective function
    if obj_type=="azp"
        @objective(model, Min, (1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt))
    elseif obj_type =="scc"
        @objective(model, Min, (1/nt)*sum(sum(-scc_weights[i]*(σ⁺[i, k] + σ⁻[i, k]) for i ∈ 1:np) for k ∈ 1:nt))
    elseif obj_type == "azp-scc"
        f1a=f_data.f1a
        f1b=f_data.f1b
        f2a=f_data.f2a
        f2b=f_data.f2b
        @objective(model, Min, 
            (1-ω)*((((1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt)) - f1a) / (f1b - f1a)) + ω*((((1/nt)*sum(sum(-scc_weights[j]*(σ⁺[j, k] + σ⁻[j, k]) for j ∈ 1:np) for k ∈ 1:nt)) - f2a) / (f2b - f2a)))
    end
    
    optimize!(model)
    convex_relax_sol = (q=value.(q), h=value.(h), η=value.(η), α=value.(α), v=value.(v), z=value.(z), y=value.(y), obj_relax=objective_value(model), cpu_time=solve_time(model))

    # return convex_relax_sol
    return value.(q), value.(h), value.(η), value.(α), value.(v), value.(z), value.(y), objective_value(model), solve_time(model)

end



function solve_convex_relax(network, probdata, obj_type; nv::Int64=0, nf::Int64=0, num_OA_cuts::Int64=0, ω=nothing, f_data=nothing, network_decomp=nothing)

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

    # unload problem data
    q_lo = copy(probdata.Qmin)
    q_up = copy(probdata.Qmax)
    h_lo = copy(probdata.Hmin)
    h_up = copy(probdata.Hmax)
    η_lo = copy(probdata.ηmin)
    η_up = copy(probdata.ηmax)
    α_up = copy(probdata.αmax)
    α_lo = zeros(nn, nt)

     # define HW head loss model parameters
     r = determine_r(network.L, network.D, network.C, network.nexp, valve_idx=network.valve_idx)
     ϕ_ex(q, r, nexp) = r.*q.*abs.(q).^(nexp.-1) # define head loss model

     # implement QA model later...
 
     # assign pcv and dbv valve data
     v_lo = zeros(2*np, 1)
     v_up = zeros(2*np, 1)
     if nv > 0
        v_up = ones(2*np, 1)
    end
     if !isempty(network.pcv_loc)
         v_lo[network.pcv_idx, :] .= 1
     end
     if network_decomp !== nothing
        if obj_type == "scc" || obj_type == "azp-scc"
            v_up[vcat(network_decomp.F, network_decomp.S)] .= 0
            v_up[vcat(network_decomp.F, network_decomp.S) .+ np] .= 0
            # v_up[network_decomp.F] .= 0
            # v_up[network_decomp.F .+ np] .= 0
        else
            v_up[network_decomp.F] .= 0
            v_up[network_decomp.F .+ np] .= 0
        end
     end
 
     # AFV data
     y_lo = zeros(nn, 1)
     y_up = zeros(nn, 1)
     if nf > 0
         y_up = ones(nn, 1)
     end
     
     # set θ bounds
     θ_lo = ϕ_ex(q_lo, r, nexp)
     θ_up = ϕ_ex(q_up, r, nexp)

    # build optimization model
    model = Model(Gurobi.Optimizer)
    # set_optimizer_attribute(model,"Method", 2)
    # set_optimizer_attribute(model,"Presolve", 0)
    # set_optimizer_attribute(model,"Crossover", 0)
    # set_optimizer_attribute(model,"NumericFocus", 3)
    set_silent(model)
    
    # define variables
    @variable(model, q_lo[i, k] ≤ q[i=1:np, k=1:nt] ≤ q_up[i, k])
    @variable(model, h_lo[i, k] ≤ h[i=1:nn, k=1:nt] ≤ h_up[i, k])
    @variable(model, η_lo[i, k] ≤ η[i=1:np, k=1:nt] ≤ η_up[i, k])
    @variable(model, θ_lo[i, k] ≤ θ[i=1:np, k=1:nt] ≤ θ_up[i, k])
    @variable(model, α_lo[i, k] ≤ α[i=1:nn, k=1:nt] ≤ α_up[i, k])
    @variable(model, v_lo[i] ≤ v[i=1:2*np] ≤ v_up[i])
    @variable(model, y_lo[i] ≤ y[i=1:nn] ≤ y_up[i])

    # hydraulic constraints
    @constraint(model, θ + A12*h + A10*h0 + η .== 0)
    @constraint(model, A12'*q - α .== d)

    # big-M valve contraints
    @constraint(model, [i=1:np, k=1:nt], η[i, k] -η_up[i, k]*v[i] ≤ 0)
    @constraint(model, [i=1:np, k=1:nt], -η[i, k] +η_lo[i, k]*v[np+i] ≤ 0)
    @constraint(model, [i=1:np, k=1:nt], -q[i, k] -q_lo[i, k]*v[i] ≤ -q_lo[i, k])
    @constraint(model, [i=1:np, k=1:nt], q[i, k] +q_up[i, k]*v[np+i] ≤ q_up[i, k])
    @constraint(model, [i=1:np, k=1:nt], -θ[i, k] -θ_lo[i, k]*v[i] ≤ -θ_lo[i, k])
    @constraint(model, [i=1:np, k=1:nt], θ[i, k] +θ_up[i, k]*v[np+i] ≤ θ_up[i, k])
    @constraint(model, [i=1:nn, k=1:nt], α[i, k] -α_up[i, k]*y[i] ≤ 0)

    # valve placement constraints
    @constraint(model, sum(v) == nv)
    @constraint(model,[i=1:np], v[i]+v[np+i] ≤ 1)
    @constraint(model, sum(y) == nf)
    
    # head loss model relaxation constraints
    R, E, R_rhs = friction_HW_relax(q_lo, q_up, r, nexp)
    @constraint(model, ϕ_relax1[i=1:np, k=1:nt], R[i, k, 1] * q[i, k] + E[i, k, 1] * θ[i, k] ≤ R_rhs[i, k, 1] )
    @constraint(model, ϕ_relax2[i=1:np, k=1:nt], R[i, k, 2] * q[i, k] + E[i, k, 2] * θ[i, k] ≤ R_rhs[i, k, 2] )
    @constraint(model, ϕ_relax3[i=1:np, k=1:nt], R[i, k, 3] * q[i, k] + E[i, k, 3] * θ[i, k] ≤ R_rhs[i, k, 3] )
    @constraint(model, ϕ_relax4[i=1:np, k=1:nt], R[i, k, 4] * q[i, k] + E[i, k, 4] * θ[i, k] ≤ R_rhs[i, k, 4] )

    # additional model terms for SCC optimization problem
    if obj_type == "scc" || obj_type == "azp-scc"
        # sigmoid function auxiliary variables
        @variable(model, 0 ≤ σ⁺[i=1:network.np, k=1:network.nt] ≤ 1)
        @variable(model, 0 ≤ σ⁻[i=1:network.np, k=1:network.nt] ≤ 1)

        # sigmoid function relaxation constraints
        pipe_csa =  1000*(π * (network.D) .^ 2) / 4
        S, T, S_rhs = sigmoid_relax(q_lo, q_up, opt_params.umax, opt_params.umin, pipe_csa, opt_params.ρ)
        @constraint(model, ψ⁺_relax1[i=1:np, k=1:nt], S[i, k, 1] * q[i, k] + T[i, k, 1] * σ⁺[i, k] ≤ S_rhs[i, k, 1] )
        @constraint(model, ψ⁺_relax2[i=1:np, k=1:nt], S[i, k, 2] * q[i, k] + T[i, k, 2] * σ⁺[i, k] ≤ S_rhs[i, k, 2] )
        @constraint(model, ψ⁻_relax1[i=1:np, k=1:nt], S[i, k, 3] * q[i, k] + T[i, k, 3] * σ⁻[i, k] ≤ S_rhs[i, k, 3] )
        @constraint(model, ψ⁻_relax2[i=1:np, k=1:nt], S[i, k, 4] * q[i, k] + S[i, k, 4] * σ⁻[i, k] ≤ S_rhs[i, k, 4] )
    end
    
    # assign model objective function
    if obj_type=="azp"
        @objective(model, Min, (1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt))
    elseif obj_type =="scc"
        @objective(model, Min, (1/nt)*sum(sum(-scc_weights[i]*(σ⁺[i, k] + σ⁻[i, k]) for i ∈ 1:np) for k ∈ 1:nt))
    elseif obj_type == "azp-scc"
        f1a=f_data.f1a
        f1b=f_data.f1b
        f2a=f_data.f2a
        f2b=f_data.f2b
        @objective(model, Min, 
            (1-ω)*((((1/nt)*sum(sum(azp_weights[i]*(h[i, k] - elev[i]) for i ∈ 1:nn) for k ∈ 1:nt)) - f1a) / (f1b - f1a)) + ω*((((1/nt)*sum(sum(-scc_weights[j]*(σ⁺[j, k] + σ⁻[j, k]) for j ∈ 1:np) for k ∈ 1:nt)) - f2a) / (f2b - f2a)))
    end
    
    optimize!(model)

    # return convex_relax_sol
    return value.(q), value.(h), value.(η), value.(α), value.(v), value.(y), objective_value(model), solve_time(model)

end



function friction_HW_relax(qminMat, qmaxMat, r, nexpvec)

    np = size(qminMat, 1)
    nt = size(qminMat, 2)
    R1vec = zeros(np*nt)
    r1 = zeros(np*nt)
    R2vec = zeros(np*nt)
    r2 = zeros(np*nt)
    R3vec = zeros(np*nt)
    r3 = zeros(np*nt)
    R4vec = zeros(np*nt)
    r4 = zeros(np*nt)

    qmin = qminMat[:]
    qmax = qmaxMat[:]
    nexp = repeat(nexpvec, nt, 1)
    r = repeat(r, nt, 1)
    
    
    for i = 1:np*nt
        ql = qmin[i]
        qu = qmax[i]
        n = nexp[i]
        r_i = r[i]
        f(q) = r_i*q*abs(q)^(n-1)
        df(q) = n*r_i*abs(q)^(n-1)
        z1, z2 = compute_HW_flexpoint(ql, qu, n)

        if (ql<0) && (qu>0) && !isnan(z1) && !isnan(z2)
        
            mpos = (f(z2)-f(ql))/(z2-ql)
            mneg = (f(z1)-f(qu))/(z1-qu)
            R1vec[i] = df(qu)
            r1[i] = -(f(qu)-df(qu)*qu)
            R2vec[i] = -df(ql)
            r2[i] = f(ql)-df(ql)*ql
            R3vec[i] = mpos
            r3[i] = -(f(ql)-mpos*ql)
            R4vec[i] = -mneg
            r4[i] = f(qu)-mneg*qu       
        
        end
        if  (ql < 0) &&  (qu > 0) && !isnan(z1) && isnan(z2)
            
            mneg = (f(z1)-f(qu))/(z1-qu)
            mLU = (f(ql)-f(qu))/(ql-qu)
            R1vec[i] = NaN
            r1[i] = NaN
            R2vec[i] = -df(ql)
            r2[i] = f(ql)-df(ql)*ql
            R3vec[i] = mLU
            r3[i] = -(f(ql)-mLU*ql)
            R4vec[i] = -mneg
            r4[i] = f(qu)-mneg*qu
        
        end
        if  (ql < 0) &&  (qu > 0) && isnan(z1) & !isnan(z2)

            mpos = (f(z2)-f(ql))/(z2-ql)
            mLU = (f(ql)-f(qu))/(ql-qu)
            R1vec[i] = df(qu)
            r1[i] = -(f(qu)-df(qu)*qu)
            R2vec[i] = NaN
            r2[i] = NaN
            R3vec[i] = mpos
            r3[i] = -(f(ql)-mpos*ql)
            R4vec[i] = -mLU
            r4[i] = f(ql)-mLU.*ql
                 
        end
        if (ql >= 0) && (qu >= 0) && (ql != qu)

            mLU = (f(ql)-f(qu))/(ql-qu)
            R1vec[i] = df(qu)
            r1[i] = -(f(qu)-df(qu)*qu)
            R2vec[i] = -mLU
            r2[i] = f(ql)-mLU*ql
            R3vec[i] = df(ql)
            r3[i] = -(f(ql)-df(ql)*ql)
            R4vec[i] = NaN
            r4[i] = NaN
            
        end

        if (ql <=0) && (qu <= 0) && (ql != qu)
            
            mLU = (f(ql)-f(qu))/(ql-qu)
            R1vec[i] = mLU
            r1[i] = -(f(ql)-mLU*ql)
            R2vec[i] = -df(qu)
            r2[i] = f(qu)-df(qu)*qu
            R3vec[i] = NaN
            r3[i] = NaN
            R4vec[i] = -df(ql)
            r4[i] = f(ql)-df(ql)*ql
        
        end
        if ql == qu

            R1vec[i] = 0
            r1[i] = -f(ql)
            R2vec[i] = 0
            r2[i] = f(ql)
            R3vec[i] = NaN
            r3[i] = NaN
            R4vec[i] = NaN
            r4[i] = NaN
            
        end
    end

    E1vec = -ones(np*nt)
    E2vec = ones(np*nt)
    E3vec = -ones(np*nt)
    E4vec = ones(np*nt)
     
    J1 = isnan.(R1vec)
    J2 = isnan.(R2vec)
    J3 = isnan.(R3vec)
    J4 = isnan.(R4vec)

    R1vec[J1] .= 0
    R2vec[J2] .= 0
    R3vec[J3] .= 0
    R4vec[J4] .= 0
    E1vec[J1] .= 0
    E2vec[J2] .= 0
    E3vec[J3] .= 0
    E4vec[J4] .= 0
    r1[J1] .= 0
    r2[J2] .= 0
    r3[J3] .= 0
    r4[J4] .= 0

    R = zeros(np, nt, 4)
    E = zeros(np, nt, 4)
    r = zeros(np, nt, 4)
    R[:, :, 1] = reshape(R1vec, np, nt)
    R[:, :, 2] = reshape(R2vec, np, nt)
    R[:, :, 3] = reshape(R3vec, np, nt)
    R[:, :, 4] = reshape(R4vec, np, nt)
    E[:, :, 1] = reshape(E1vec, np, nt)
    E[:, :, 2] = reshape(E2vec, np, nt)
    E[:, :, 3] = reshape(E3vec, np, nt)
    E[:, :, 4] = reshape(E4vec, np, nt)
    r[:, :, 1] = reshape(r1, np, nt)
    r[:, :, 2] = reshape(r2, np, nt)
    r[:, :, 3] = reshape(r3, np, nt)
    r[:, :, 4] = reshape(r4, np, nt)

    return R, E, r

end



function compute_HW_flexpoint(a, b, n)
    
    if (a<0)&&(b>0)&&(b-a>0)
        g1(x) = (n-1)*(abs(x).^(n-1)).*x - n*b*abs(x).^(n-1) + b*abs(b)^(n-1)
        #Check if there is a zero in [a,0]
        if g1(a)*g1(0)<0
            xlo = a
            xup = 0
            xm = (xlo+xup)/2
            while abs(xup-xlo)>=1e-12
                if g1(xm)*g1(xlo)<0
                    xup = xm
                    xm = (xlo+xup)/2
                else
                    xlo = xm
                    xm = (xlo+xup)/2
                end
            end
            z1 = xm
        else
            z1 = NaN
        end
        
        g2(x) = (n-1)*(abs(x).^(n-1)).*x - n*a*abs(x).^(n-1) + a*abs(a)^(n-1)
        # Check there exist a zero in [0,b]:
        if g2(b)*g2(0)<0
            xlo = 0
            xup = b
            xm = (xlo+xup)/2
            while abs(xup-xlo)>=1e-12
                if g2(xm)*g2(xlo)<0                
                    xup = xm
                    xm = (xlo+xup)/2
                else
                    xlo = xm
                    xm = (xlo+xup)/2
                end
            end
            z2= xm          
        else
            z2=NaN 
        end 
            
    else
        z1=NaN
        z2=NaN        
    end

    return z1,z2

end



function sigmoid_relax(qminMat, qmaxMat, umax, umin, pipe_csa, ρ)
    
    # initialize data
    np = size(qminMat, 1)
    nt = size(qminMat, 2)
    S1vec_pos = zeros(np*nt)
    s1_pos = zeros(np*nt)
    S2vec_pos = zeros(np*nt)
    s2_pos = zeros(np*nt)
    S1vec_neg = zeros(np*nt)
    s1_neg = zeros(np*nt)
    S2vec_neg = zeros(np*nt)
    s2_neg = zeros(np*nt)

    umax = repeat(umax, nt, 1)
    pipe_csa = repeat(pipe_csa, nt, 1)
    u_lo = qminMat[:] ./ pipe_csa
    u_up = qmaxMat[:] ./ pipe_csa

    # define functions
    σ(u, β) = (1+exp(-ρ*(β*u-umin)))^-1
    ∇σ(u, β) = β*ρ*(exp(-ρ*(β*u-umin))*((1+exp(-ρ*(β*u-umin)))^-2))

    # assign relaxation parameters
    for i = collect(1:np*nt)
        tu_lo = u_lo[i]
        tu_up = u_up[i]
        w⁺, w⁻ = compute_sigmoid_flexpoint(tu_lo, tu_up, umin, umax[i], ρ)

        """
            Linear relaxations for positive sigmoid function
        """
        # Case 1: LB < w & UB > w
        if (tu_lo < w⁺) && (tu_up > w⁺)
            S1vec_pos[i] = ∇σ(w⁺, 1) / pipe_csa[i]
            s1_pos[i] = σ(w⁺, 1) - ∇σ(w⁺, 1) * w⁺
            S2vec_pos[i] = ∇σ(tu_up, 1) / pipe_csa[i]
            s2_pos[i] = σ(tu_up, 1) - ∇σ(tu_up, 1) * tu_up

        # Case 2: LB < w & UB < w & vlo != vup
        elseif (tu_lo < w⁺) && (tu_up < w⁺) && (tu_lo != tu_up)
            S1vec_pos[i] = ((σ(tu_up, 1) - σ(tu_lo, 1)) / (tu_up - tu_lo)) / pipe_csa[i]
            s1_pos[i] = σ(tu_lo, 1) - (σ(tu_up, 1) - σ(tu_lo, 1)) / (tu_up - tu_lo) * tu_lo
            S2vec_pos[i] = NaN
            s2_pos[i] = NaN

        # Case 3: LB = w & UB > w
        elseif (tu_lo == w⁺) && (tu_up > w⁺)
            S1vec_pos[i] = ∇σ(tu_lo, 1) / pipe_csa[i]
            s1_pos[i] = σ(tu_lo, 1) - ∇σ(tu_lo, 1) * w⁺
            S2vec_pos[i] = ∇σ(tu_up, 1) / pipe_csa[i]
            s2_pos[i] = σ(tu_up, 1) - ∇σ(tu_up, 1) * tu_up

        # Case 4: LB = UB
        elseif (tu_lo == tu_up)
            S1vec_pos[i] = 0
            s1_pos[i] = σ(w⁺, 1)
            S2vec_pos[i] = NaN
            s2_pos[i] = NaN

        end

        """
            Linear relaxations for negative sigmoid function
            NB: LB is u_up and vice-versa for negative sigmoid case
        """
        # Case 1: LB < w & UB > w
        if (tu_up > w⁻) && (tu_lo < w⁻)
            S1vec_neg[i] = ∇σ(w⁻, -1) / pipe_csa[i]
            s1_neg[i] = σ(w⁻, -1) - ∇σ(w⁻, -1) * w⁻
            S2vec_neg[i] = ∇σ(tu_lo, -1) / pipe_csa[i]
            s2_neg[i] = σ(tu_lo, -1) - ∇σ(tu_lo, -1) * tu_lo

        # Case 2: LB < w & UB < w & vlo != vup
        elseif (tu_up > w⁻) && (tu_lo > w⁻) && (tu_lo != tu_up)
            S1vec_neg[i] = ((σ(tu_lo, -1) - σ(tu_up, -1)) / (tu_lo - tu_up)) / pipe_csa[i]
            s1_neg[i] = σ(tu_up, -1) - (σ(tu_lo, -1) - σ(tu_up, -1)) / (tu_lo - tu_up) * tu_up
            S2vec_neg[i] = NaN
            s2_neg[i] = NaN       

        # Case 3: LB = w & UB > w
        elseif (tu_up == w⁻) && (tu_lo < w⁻)
            S1vec_neg[i] = ∇σ(tu_up, -1) / pipe_csa[i]
            s1_neg[i] = σ(tu_up, -1) - ∇σ(tu_up, -1) * w⁻
            S2vec_neg[i] = ∇σ(tu_lo, -1) / pipe_csa[i]
            s2_neg[i] = σ(tu_lo, -1) - ∇σ(tu_lo, -1) * tu_lo

        # Case 4: LB = UB
        elseif (tu_lo == tu_up)
            S1vec_neg[i] = 0
            s1_neg[i] = σ(w⁻, 1)
            S2vec_neg[i] = NaN
            s2_neg[i] = NaN
            
        end
    end

    # ignore constraints with NaN values
    T1vec_pos = ones(np*nt)
    T2vec_pos = ones(np*nt)
    T1vec_neg = ones(np*nt)
    T2vec_neg = ones(np*nt)
     
    J1 = isnan.(S1vec_pos)
    J2 = isnan.(S2vec_pos)
    J3 = isnan.(S1vec_neg)
    J4 = isnan.(S2vec_neg)

    S1vec_pos[J1] .= 0
    S2vec_pos[J2] .= 0
    S1vec_neg[J3] .= 0
    S2vec_neg[J4] .= 0
    T1vec_pos[J1] .= 0
    T2vec_pos[J2] .= 0
    T1vec_neg[J3] .= 0
    T2vec_neg[J4] .= 0
    s1_pos[J1] .= 0
    s2_pos[J2] .= 0
    s1_neg[J3] .= 0
    s2_neg[J4] .= 0

    # form linear relaxation coefficients
    S = zeros(np, nt, 4)
    T = zeros(np, nt, 4)
    s = zeros(np, nt, 4)
    S[:, :, 1] = reshape(-S1vec_pos, np, nt)
    S[:, :, 2] = reshape(-S2vec_pos, np, nt)
    S[:, :, 3] = reshape(-S1vec_neg, np, nt)
    S[:, :, 4] = reshape(-S2vec_neg, np, nt)
    T[:, :, 1] = reshape(T1vec_pos, np, nt)
    T[:, :, 2] = reshape(T2vec_pos, np, nt)
    T[:, :, 3] = reshape(T1vec_neg, np, nt)
    T[:, :, 4] = reshape(T2vec_neg, np, nt)
    s[:, :, 1] = reshape(s1_pos, np, nt)
    s[:, :, 2] = reshape(s2_pos, np, nt)
    s[:, :, 3] = reshape(s1_neg, np, nt)
    s[:, :, 4] = reshape(s2_neg, np, nt)

    return S, T, s


end


"""

compute_sigmoid_flexpoint
    - compute flex point for sigmoid relaxation, which forms a concave envelope (outer approximation)
    - flex point (w) is the root of intersection between tangeant of σ at w and σ(w)
    - lower bound (L): σ(L) = σ(w) + ∇σ(L-w)

"""


function compute_sigmoid_flexpoint(u_lo, u_up, umin, umax, ρ)

    # define functions
    σ(u, β) = (1+exp(-ρ*(β*u-umin)))^-1
    ∇σ(u, β) = β*ρ*(exp(-ρ*(β*u-umin))*((1+exp(-ρ*(β*u-umin)))^-2))
    f(u, u_bound, β) = ∇σ(u, β)*(u_bound-u) - σ(u_bound, β) + σ(u, β) # equation of intersection

    β = [1, -1]
    w = zeros(2)

    ### compute positive and negative flex points
    for (i, b) in enumerate(β)
        if b == 1
            u_bound = u_lo
            x_up = umax
            x_lo = umin
        else
            u_bound = u_up
            x_up = -umin # starting point at sigmoid inflection point (ensures w is on concave part of curve)
            x_lo = -umax
        end

        if b*u_bound > umin
            w[i] = u_bound
        else
            x_m = (x_lo+x_up)/2

            while abs.(x_up-x_lo) ≥ 1e-14
                if f(x_m, u_bound, b)*f(x_up, u_bound, b) < 0
                    x_lo = x_m
                    x_m = (x_lo+x_up)/2
                else
                    x_up = x_m
                    x_m = (x_lo+x_up)/2
                end
            end
            w[i] = x_m
        end
    end

    return w[1], w[2] # positive flex point, negative flex point

end




function random_sampling(network, probdata, obj_type, N, nv, nf, ω, f_data, nlp_solver, η_init, bi_dir, v_idx, v_prob, y_idx, y_prob)

    # initialize empty arrays
    nv_select = []
    nf_select = []
    valve_removed = []
    valve_infeas = []
    valve_visited = []
    x_sol = Array{Union{Nothing, Float64}}(nothing, network.nt*(network.np+network.nn+network.np+network.nn), N)
    obj_sol = Array{Union{Nothing, Float64}}(nothing, N)
    cpu_sol = Array{Union{Nothing, Float64}}(nothing, N)

    # set binary values and solve NLP problem for each trial i ∈ {1,…,N}
    @showprogress for i ∈ collect(1:N)

        bool1 = true
        bool2 = true
        feas = false
        x_init = []
        v_loc = []
        y_loc = []
        probdata_nlp = copy(probdata)

        while isempty(hcat(nv_select', nf_select')) || any((bool1, bool2, !feas))

            # weighted random sampling
            nv_select = sample(v_idx, Weights(v_prob), nv,  replace=false)
            nf_select = sample(y_idx, Weights(y_prob), nf,  replace=false)

            # check if new sample is unique
            if i > 1
                for row ∈ collect(1:i-1)
                    bool1 = all(x -> x ∈ valve_visited[row], hcat(nv_select', nf_select'))
                    if bool1 
                        break # resample to find unique valve configuration
                    end
                end
            else
                bool1 = false
            end

            # check if PCVs are feasible
            n = size(valve_infeas, 1)
            if n > 0
                for row ∈ collect(1:n)
                    bool2 = all(x -> x ∈ valve_infeas[row], nv_select')
                    if bool2 
                        break # resample to find unique valve configuration
                    end
                end
            else
                bool2 = false
            end


            if !bool1 && !bool2

                # assign control valve configuration
                if nv > 0 && nf > 0
                    v_loc = nv_select
                    y_loc = nf_select
                elseif nv == 0 && nf > 0
                    v_loc = nothing
                    y_loc = nf_select
                elseif nv > 0 && nf == 0
                    v_loc = nv_select
                    y_loc = nothing
                end

                # re-initialise probdata
                probdata_nlp = copy(probdata)

                # assign afv bounds
                if !isnothing(y_loc)
                    probdata_nlp.αmax[setdiff(1:network.nn, y_loc), :] .= 0
                else
                    probdata_nlp.αmax .= 0
                end

                # assign pcv bounds
                if bi_dir
                    A13 = spzeros(network.np)
                    A13[v_loc] .= 1
                    probdata_nlp.ηmin[setdiff(1:network.np, v_loc), :] .= 0
                    probdata_nlp.ηmax[setdiff(1:network.np, v_loc), :] .= 0
                else
                    A13 = spzeros(network.np)
                    for (i, v) ∈ enumerate(v_loc)
                        if v ≤ network.np
                            for k ∈ collect(1:network.nt)
                                probdata_nlp.ηmin[v, k] = maximum([0, probdata_nlp.ηmin[v, k]])
                                probdata_nlp.Qmin[v, k] = maximum([0, probdata_nlp.Qmin[v, k]])
                                if η_init[v, k] < 0
                                    η_init[v, k] = 0
                                end
                            end
                            A13[v] = 1
                        else
                            v = v - network.np
                            for k ∈ collect(1:network.nt)
                                probdata_nlp.ηmax[v, k] = minimum([0, probdata_nlp.ηmax[v, k]])
                                probdata_nlp.Qmax[v, k] = minimum([0, probdata_nlp.Qmax[v, k]])
                                if η_init[v, k] > 0
                                    η_init[v, k] = 0
                                end
                            end
                            A13[v] = 1
                            v_loc[i] = Int(v)
                        end
                    end
                    probdata_nlp.ηmin[setdiff(1:network.np, v_loc), :] .= 0
                    probdata_nlp.ηmax[setdiff(1:network.np, v_loc), :] .= 0
                end

                # check hydraulic feasibility
                if bi_dir
                    x_init, feas = make_starting_point(network, probdata_nlp, obj_type, η_init, v_loc, y_loc, A13, bi_dir)
                else
                    infeas_Q = findall(x->any(vec(probdata_nlp.Qmax[x, :] - probdata_nlp.Qmin[x, :]) .≤ 0), v_loc)
                    if isempty(infeas_Q)
                        # all Q bounds are feasible; move to making feasible starting point
                        x_init, feas = make_starting_point(network, probdata_nlp, obj_type, η_init, v_loc, y_loc, A13, bi_dir) 
                    else
                        # at least 1 Q bound is infeasible; remove from list of valve candidates
                        push!(valve_removed, nv_select[infeas_Q])
                        idx_rm = findall(x -> x == nv_select[infeas_Q][1], v_idx)
                        deleteat!(v_idx, idx_rm)
                        deleteat!(v_prob, idx_rm)
                        feas = false
                    end
                end

                # store infeasible solutions
                if !feas
                    push!(valve_infeas, nv_select') 
                end  

            end

        end

        # store control valve configuration
        push!(valve_visited, hcat(nv_select', nf_select'))

        # solve nlp control problem
        if nlp_solver == "ipopt"
                x_sol[:, i], obj_sol[i], cpu_sol[i] = solve_nlp_ipopt(network, probdata_nlp, obj_type; v_loc=v_loc, y_loc=y_loc, f_data=f_data, ω=ω, x_init=x_init, bi_dir=bi_dir)

        elseif nlp_solver == "scp"
            # SCP CODE NEEDS TO BE UPDATED!!!

        end

    end

    return x_sol, obj_sol, cpu_sol, valve_visited

end





function round_and_swap(network, probdata, obj_type, N, nv, nf, ω, f_data, nlp_solver, η_init, bi_dir, v_idx, v_prob, y_idx, y_prob)

    # initialize data
    np = network.np
    nn = network.nn
    x_sol = Array{Union{Nothing, Float64}}(nothing, network.nt*(network.np+network.nn+network.np+network.nn), N)
    obj_sol = Array{Union{Nothing, Float64}}(nothing, N)
    cpu_sol = Array{Union{Nothing, Float64}}(nothing, N)
    iter = 1
    swap = 0
    bi_dir = true # keep true for now... 
    prog = Progress(N)
    valve_infeas = []
    valve_visited = []
    bool1 = false
    bool2 = false

    # set v vector data
    v_data = hcat(v_idx, v_prob)
    v_loc = map(Int64, v_data[1:nv, 1])
    v_hat = v_data[1:nv, 2]
    v_sample = v_data[findall(x -> x ∉ v_loc, v_data[:, 1]), :]
    v_test = v_loc

    # set y vector data
    y_data = hcat(y_idx, y_prob)
    y_loc = map(Int64, y_data[1:nf, 1])
    y_hat = y_data[1:nf, 2]
    y_sample = y_data[findall(x -> x ∉ y_loc, y_data[:, 1]), :]
    y_test = y_loc

    # find first feasible solution
    probdata_nlp = copy(probdata)
    A13 = spzeros(np)
    A13[v_loc] .= 1
    probdata_nlp.ηmin[setdiff(1:np, v_loc), :] .= 0
    probdata_nlp.ηmax[setdiff(1:np, v_loc), :] .= 0
    probdata_nlp.αmax[setdiff(1:nn, y_loc), :] .= 0

    x_init, feas = make_starting_point(network, probdata_nlp, obj_type, η_init, v_loc, y_loc, A13, bi_dir)

    if feas
        if nlp_solver == "ipopt"
            x_sol[:, iter], obj_sol[iter], cpu_sol[iter] = solve_nlp_ipopt(network, probdata_nlp, obj_type; v_loc=v_loc, y_loc=y_loc, f_data=f_data, ω=ω, x_init=x_init, bi_dir=bi_dir)
        elseif nlp_solver == "scp"
            # SCP CODE NEEDS TO BE UPDATED!!!
        end
        best_sol = obj_sol[1]
        iter += 1
        push!(valve_visited, hcat(v_loc', y_loc')) 
        next!(prog)
    else
        best_sol = Inf
        push!(valve_infeas, v_loc') 
    end

    # swapping routine
    while iter ≤ N
        flag = 0
        i = 0

        to_remove = vcat(v_hat, y_hat)

        while i ≤ nv+nf # outer loop to remove valve locations
            i += 1

            ### Step 1: find lowest fractional value to remove ###
            rm_idx = argmin(to_remove)
            to_remove[rm_idx] = Inf
            v_test = v_loc
            y_test = y_loc

            if rm_idx ≤ nv
                for (idx, v) ∈ enumerate(eachrow(v_data[findall(x -> x ∉ v_test, v_data[:, 1]), :])) # inner for loop to replace pcv location
                    flag = 0
                    v_test[rm_idx] = v[1]

                    if iter > 1
                        for row ∈ collect(1:iter-1)
                            bool1 = all(x -> x ∈ valve_visited[row], hcat(v_test', y_loc'))
                            if bool1 
                                break
                            end
                        end
                        if bool1
                            flag = 2 # find unique valve configuration
                        end
                    end
        
                    # check if PCVs are feasible
                    n = size(valve_infeas, 1)
                    if n > 0
                        for row ∈ collect(1:n)
                            bool2 = all(x -> x ∈ valve_infeas[row], v_test')
                            if bool2 
                                break
                            end
                        end
                        if bool2
                            flag = 2 # find unique valve configuration
                        end
                    end

                    if flag != 2
                        # find feasible solution
                        probdata_nlp = copy(probdata)
                        A13 = spzeros(np)
                        A13[v_test] .= 1
                        probdata_nlp.ηmin[setdiff(1:np, v_test), :] .= 0
                        probdata_nlp.ηmax[setdiff(1:np, v_test), :] .= 0
                        probdata_nlp.αmax[setdiff(1:nn, y_loc), :] .= 0

                        x_init, feas = make_starting_point(network, probdata_nlp, obj_type, η_init, v_test, y_loc, A13, bi_dir)

                        if feas
                            if nlp_solver == "ipopt"
                                x_sol[:, iter], obj_sol[iter], cpu_sol[iter] = solve_nlp_ipopt(network, probdata_nlp, obj_type; v_loc=v_test, y_loc=y_loc, f_data=f_data, ω=ω, x_init=x_init, bi_dir=bi_dir)
                            elseif nlp_solver == "scp"
                                # SCP CODE NEEDS TO BE UPDATED!!!
                            end

                            # check if feasible solution is best
                            if obj_sol[iter] < best_sol
                                best_sol = obj_sol[iter]
                                v_loc = v_test
                                v_hat[rm_idx] = v[2]
                                swap += 1
                                flag = 1
                            end

                            push!(valve_visited, hcat(v_test', y_loc')) 

                            iter += 1
                            next!(prog)

                        else
                            push!(valve_infeas, v_test') 

                        end

                        if iter > N
                            flag = 1
                        end

                        if flag == 1
                            break
                        end
                    end
                end

            else
                rm_idx -= nv
                for (idx, y) ∈ enumerate(eachrow(y_data[findall(x -> x ∉ y_test, y_data[:, 1]), :])) # inner for loop to replace pcv location
                    flag = 0
                    y_test[rm_idx] = y[1]

                    if iter > 1
                        for row ∈ collect(1:iter-1)
                            bool1 = all(x -> x ∈ valve_visited[row], hcat(v_loc', y_test'))
                            if bool1 
                                break
                            end
                        end
                        if bool1
                            flag = 2 # find unique valve configuration
                        end
                    end
        
                    # check if PCVs are feasible
                    n = size(valve_infeas, 1)
                    if n > 0
                        for row ∈ collect(1:n)
                            bool2 = all(x -> x ∈ valve_infeas[row], v_loc')
                            if bool2 
                                break 
                            end
                        end
                        if bool2
                            break
                        end
                    end

                    if flag != 2
                        # find feasible solution
                        probdata_nlp = copy(probdata)
                        A13 = spzeros(np)
                        A13[v_test] .= 1
                        probdata_nlp.ηmin[setdiff(1:np, v_loc), :] .= 0
                        probdata_nlp.ηmax[setdiff(1:np, v_loc), :] .= 0
                        probdata_nlp.αmax[setdiff(1:nn, y_test), :] .= 0

                        x_init, feas = make_starting_point(network, probdata_nlp, obj_type, η_init, v_loc, y_test, A13, bi_dir)

                        if feas
                            if nlp_solver == "ipopt"
                                x_sol[:, iter], obj_sol[iter], cpu_sol[iter] = solve_nlp_ipopt(network, probdata_nlp, obj_type; v_loc=v_loc, y_loc=y_test, f_data=f_data, ω=ω, x_init=x_init, bi_dir=bi_dir)
                            elseif nlp_solver == "scp"
                                # SCP CODE NEEDS TO BE UPDATED!!!
                            end

                            # check if feasible solution is best
                            if obj_sol[iter] < best_sol
                                best_sol = obj_sol[iter]
                                y_loc = y_test
                                y_hat[rm_idx] = y[2]
                                swap += 1
                                flag = 1
                            end

                            push!(valve_visited, hcat(v_loc', y_test')) 

                            iter += 1
                            next!(prog)

                        else
                            push!(valve_infeas, v_loc') 

                        end

                        if iter > N
                            flag = 1
                        end

                        if flag == 1
                            break
                        end
                    end

                end
            end

            if flag == 1
                break
            end

        end

        if flag == 0 || i > nf+nv
            break
        end


    end

    return x_sol, obj_sol, cpu_sol, valve_visited

end
