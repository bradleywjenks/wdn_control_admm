### Optimization-based bound tightening (OBBT) functions
using FLoops
using LinearAlgebra
using SparseArrays


function run_obbt(network, opt_params; nv::Int64=0, nf::Int64=0, num_OA_cuts::Int64=0, par=false, max_iter=10, network_decomp=nothing)

    # create empty array to store tightened q_lo and q_up
    q_L = Array{Float64}(undef, network.np, network.nt, max_iter+1)
    q_U = Array{Float64}(undef, network.np, network.nt, max_iter+1)
    q_lo = copy(opt_params.Qmin)
    q_up= copy(opt_params.Qmax)

    # assign pipe list for OBBT algorithm
    if network_decomp !== nothing
        if nf > 0
            pipe_list = vcat(network_decomp.P, network_decomp.F)
        else
            pipe_list = network_decomp.P
            q_lo[network_decomp.F, :] = network_decomp.QF
            q_up[network_decomp.F, :] = network_decomp.QF
        end
    else
        pipe_list = collect(1:network.np)
    end


    # Old code (IGNORE!!!)
    # if nf > 0
    #     for (idx, f) ∈ enumerate(network_decomp.F)
    #         if sign(network_decomp.QF[idx, 1]) == 1
    #             q_lo[f, :] = network_decomp.QF[idx, :]
    #             q_up[f, :] = network_decomp.QF[idx, :] .+ opt_params.αmax[1, 1]
    #         else
    #             q_lo[f, :] = network_decomp.QF[idx, :] .- opt_params.αmax[1, 1]
    #             q_up[f, :] = network_decomp.QF[idx, :]
    #         end
    #     end
    # else
    #     q_lo[network_decomp.F, :] = network_decomp.QF
    #     q_up[network_decomp.F, :] = network_decomp.QF
    # end

    obbt_data = copy(opt_params)
    obbt_data.Qmin, obbt_data.Qmax = q_lo, q_up
    diam_ratio = 0
    diam_new = 0
    iter = 0
    tq_lo = copy(q_lo)
    tq_up = copy(q_up)

    while (iter ≤ max_iter) && (diam_ratio ≤ 0.95)
        iter +=1
        diam_old = norm(vec(tq_up .- tq_lo), Inf)
        if par
            Threads.@threads for k ∈ collect(1:network.nt)
                @show "" k Threads.threadid()
                tq_lo[:, k], tq_up[:, k] = obbt_model(network, obbt_data, k, nv, nf, pipe_list)
            end
        else
            for k ∈ collect(1:network.nt)
                tq_lo[:, k], tq_up[:, k] = obbt_model(network, obbt_data, k, nv, nf, pipe_list)
            end
        end

        diam_new = norm(vec(tq_up .- tq_lo), Inf)
        diam_ratio = diam_new/diam_old
        @info "OBBT progress: $diam_ratio"
        q_L[:, :, iter] = tq_lo
        q_U[:, :, iter] = tq_up
        obbt_data.Qmin, obbt_data.Qmax = tq_lo, tq_up
    end

    return q_L[:, :, collect(1:iter)], q_U[:, :, collect(1:iter)], iter

end


function obbt_model(network, obbt_data, k, nv, nf, pipe_list)
    q_lo = obbt_data.Qmin[:, k]
    q_up = obbt_data.Qmax[:, k]
    h_lo = obbt_data.Hmin[:, k]
    h_up = obbt_data.Hmax[:, k]
    η_lo = obbt_data.ηmin[:, k]
    η_up = obbt_data.ηmax[:, k]
    α_up = obbt_data.αmax[:, k]
    
    nexp = network.nexp
    d = network.d[:, k]
    h0 = network.h0[:, k]
    A10 = network.A10
    A12 = network.A12
    np, nn = size(A12)
    α_lo = zeros(nn, 1)

    # define HW head loss model parameters (implement QA model later...)
    r = determine_r(network.L, network.D, network.C, network.nexp, valve_idx=network.valve_idx)
    ϕ_ex(q, r, nexp) = r.*q.*abs.(q).^(nexp.-1) # head loss model

    # PCV and DBV data
    v_lo = zeros(2*np, 1)
    v_up = ones(2*np, 1)
    if !isempty(network.pcv_loc)
        v_lo[network.pcv_idx] .= 1
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
    model = Model(()->Gurobi.Optimizer(Gurobi.Env()))
    #set_optimizer_attribute(model,"Method",2)
    set_optimizer_attribute(model,"Presolve", 0)
    set_optimizer_attribute(model,"Crossover", 0)
    set_silent(model)
    
    # define variables
    @variable(model, q_lo[i] ≤ q[i=1:np] ≤ q_up[i])
    @variable(model, h_lo[i] ≤ h[i=1:nn] ≤ h_up[i])
    @variable(model, η_lo[i] ≤ η[i=1:np] ≤ η_up[i])
    @variable(model, θ_lo[i] ≤ θ[i=1:np] ≤ θ_up[i])
    @variable(model, α_lo[i] ≤ α[i=1:nn] ≤ α_up[i])
    @variable(model, v_lo[i] ≤ v[i=1:2*np] ≤ v_up[i])
    @variable(model, z_lo[i] ≤ z[i=1:np] ≤ z_up[i])
    @variable(model, y_lo[i] ≤ y[i=1:nn] ≤ y_up[i])

    # hydraulic constraints
    @constraint(model, θ + A12*h + A10*h0 + η .== 0)
    @constraint(model, A12'*q - α .== d)

    # big-M valve contraints
    @constraint(model, [i=1:np], η[i] -η_up[i]*v[i] ≤ 0)
    @constraint(model, [i=1:np], -η[i] +η_lo[i]*v[np+i] ≤ 0)
    @constraint(model, [i=1:np], -q[i] -q_lo[i]*v[i] ≤ -q_lo[i])
    @constraint(model, [i=1:np], q[i] +q_up[i]*v[np+i] ≤ q_up[i])
    @constraint(model, [i=1:np], -θ[i] -θ_lo[i]*v[i] ≤ -θ_lo[i])
    @constraint(model, [i=1:np], θ[i] +θ_up[i]*v[np+i] ≤ θ_up[i])
    @constraint(model, [i=1:nn], α[i] -α_up[i]*y[i] ≤ 0)

    # valve placement constraints
    @constraint(model, sum(z) == nv)
    @constraint(model,[i=1:np], v[i]+v[np+i] ≤ z[i])
    @constraint(model, sum(y) == nf)
    
    # head loss model relaxation constraints
    R, E, R_rhs = friction_HW_relax(q_lo, q_up, r, nexp)
    @constraint(model, ϕ_relax1[i=1:np], R[i, 1, 1] * q[i] + E[i, 1, 1] * θ[i] ≤ R_rhs[i, 1, 1] )
    @constraint(model, ϕ_relax2[i=1:np], R[i, 1, 2] * q[i] + E[i, 1, 2] * θ[i] ≤ R_rhs[i, 1, 2] )
    @constraint(model, ϕ_relax3[i=1:np], R[i, 1, 3] * q[i] + E[i, 1, 3] * θ[i] ≤ R_rhs[i, 1, 3] )
    @constraint(model, ϕ_relax4[i=1:np], R[i, 1, 4] * q[i] + E[i, 1, 4] * θ[i] ≤ R_rhs[i, 1, 4] )
    
    # if num_OA_cuts>0
    #     R_OA, E_OA, R_rhs_OA = frictionQA_add_OA_cuts(qmin,qmax,a_mat,b_mat,num_OA_cuts);
    #     @constraint(model,oacut1[i=1:np,k=1:nt,j=1:num_OA_cuts],R_OA[i,k,1,j] * q[i,k] + E_OA[i,k,1,j] * y[i,k] ≤ R_rhs_OA[i,k,1,j])
    #     @constraint(model,oacut2[i=1:np,k=1:nt,j=1:num_OA_cuts],R_OA[i,k,2,j] * q[i,k] + E_OA[i,k,2,j] * y[i,k] ≤ R_rhs_OA[i,k,2,j])
    # end

    # store results
    tq_lo = copy(q_lo)
    tq_up = copy(q_up)

    for j ∈ pipe_list
        # minimize link flow
        @objective(model, Min, q[j])
        set_start_value(q[j], tq_lo[j])
        optimize!(model)
        if termination_status(model) == OPTIMAL
            tq_lo[j] = objective_value(model)                    
        else
            tq_lo[j] = q_lo[j]
            @error "Gurobi error."
        end

        # maximize link flow
        set_objective_sense(model, MOI.MAX_SENSE) 
        set_start_value(q[j], tq_up[j])
        optimize!(model)
        if termination_status(model) == OPTIMAL
            tq_up[j] = objective_value(model)               
        else
            @error "Gurobi error."
            tq_up[j] = q_up[j]
        end

        # update q bounds
        set_lower_bound(q[j], tq_lo[j])
        set_upper_bound(q[j], tq_up[j])
        set_lower_bound(θ[j], ϕ_ex(tq_lo[j], r[j], nexp[j]))
        set_upper_bound(θ[j], ϕ_ex(tq_up[j], r[j], nexp[j]))

        # update head loss relaxation
        R_new, E_new, R_rhs_new = friction_HW_relax([tq_lo[j]], [tq_up[j]], [r[j]], [nexp[j]])

        set_normalized_coefficient(ϕ_relax1[j], q[j], R_new[1])
        set_normalized_coefficient(ϕ_relax1[j], θ[j], E_new[1])
        set_normalized_rhs(ϕ_relax1[j], R_rhs_new[1])

        set_normalized_coefficient(ϕ_relax2[j], q[j], R_new[2])
        set_normalized_coefficient(ϕ_relax2[j], θ[j], E_new[2])
        set_normalized_rhs(ϕ_relax2[j], R_rhs_new[2])

        set_normalized_coefficient(ϕ_relax3[j], q[j], R_new[3])
        set_normalized_coefficient(ϕ_relax3[j], θ[j], E_new[3])
        set_normalized_rhs(ϕ_relax3[j], R_rhs_new[3])

        set_normalized_coefficient(ϕ_relax4[j], q[j], R_new[4])
        set_normalized_coefficient(ϕ_relax4[j], θ[j], E_new[4])
        set_normalized_rhs(ϕ_relax4[j], R_rhs_new[4])

        # if num_OA_cuts>0
        #     R_OA_new, E_OA_new, R_rhs_OA_new = frictionQA_add_OA_cuts([tqmin[i,k]],[tqmax[i,k]],[a_mat[i,k]],[b_mat[i,k]],num_OA_cuts);
        #     for j=1:num_OA_cuts
        #         set_normalized_coefficient(oacut1[i,k,j], q[i,k], R_OA_new[1,1,1,j])
        #         set_normalized_coefficient(oacut1[i,k,j], y[i,k], E_OA_new[1,1,1,j])
        #         set_normalized_rhs(oacut1[i,k,j], R_rhs_OA_new[1,1,1,j])

        #         set_normalized_coefficient(oacut2[i,k,j], q[i,k], R_OA_new[1,1,2,j])
        #         set_normalized_coefficient(oacut2[i,k,j], y[i,k], E_OA_new[1,1,2,j])
        #         set_normalized_rhs(oacut2[i,k,j], R_rhs_OA_new[1,1,2,j])

        #     end
        # end
    end


    # update series link data
    for jj ∈ network_decomp.S
        # update series link q bounds
        Pjj = findall(x -> x != 0, network_decomp.E[jj, :])
        if network_decomp.E[jj, Pjj][1] == 1
            tq_lo[jj, :] = network_decomp.E[jj, Pjj][1]*tq_lo[Pjj] .+ network_decomp.E[jj, network_decomp.F]'*network_decomp.QF[:, k] .+ network_decomp.e[jj, k]
            tq_up[jj, :] = network_decomp.E[jj, Pjj][1].*tq_up[Pjj] .+ network_decomp.E[jj, network_decomp.F]'*network_decomp.QF[:, k] .+ network_decomp.e[jj, k]
        elseif network_decomp.E[jj, Pjj][1] == -1
            tq_lo[jj, :] = network_decomp.E[jj, Pjj][1].*tq_up[Pjj] .+ network_decomp.E[jj, network_decomp.F]'*network_decomp.QF[:, k] .+ network_decomp.e[jj, k]
            tq_up[jj, :] = network_decomp.E[jj, Pjj][1].*tq_lo[Pjj] .+ network_decomp.E[jj, network_decomp.F]'*network_decomp.QF[:, k] .+ network_decomp.e[jj, k]
        end
    end

    return tq_lo, tq_up

end



function forest_core_decomp(network::Network)

    # unload network variables
    np = copy(network.np)
    nn = copy(network.nn)
    nt = copy(network.nt)
    d = copy(network.d)
    A12 = copy(network.A12)
    A10 = copy(network.A10)

    P = collect(1:np) # original pipes indices
    V = collect(1:nn) # original nodes indices
    S = [] # eliminated pipes
    LOOP = [] # eliminated pipes involved in trivial loops
    Y = [] # eliminated nodes
    F = [] # eliminated forest pipes
    T = [] # eliminated forest nodes
    QSol = zeros(np, nt)
    dtilde = d
    E = speye(np, np)
    Econst = zeros(np, nt)
    
    A = hcat(A12[P, V], A10[P, :])

    ### Find forest links ###
    leaf_nodes = [node[2] for node in findall(x -> x == 1, sum(abs.(A12), dims=1))]

    while !isempty(leaf_nodes)
        j = leaf_nodes[1]
        k = nothing
        while sum(abs.(A12[:, j])) == 1
            k = findall(x -> x != 0, A12[:, j])[1]
            ids = findall(x -> x != 0, A12[k, :])
            if length(ids) == 1
                # leaf node connected to source
                QSol[k, :] = A12[k, j] .* dtilde[j, :]

                push!(T, j) # store eliminated leaf node j
                push!(F, k) # store eliminated pipe k connecting leaf node j
                A12[:, j] .= 0
                A12[k, :] .= 0
            else
                m = filter(x -> x != j, ids)[1]
                # aggregate demand
                QSol[k, :] = A12[k, j] .* dtilde[j, :] # assigns flow in pipe k equal to demand at leaf node j
                dtilde[m, :] = dtilde[m, :] + dtilde[j, :] # adds flow from leaf node to upstream node m (where j can now be cut and m is a now a leaf node)

                push!(T, j) # store eliminated leaf node j
                push!(F, k) # store eliminated pipe k connecting leaf node j
                A12[:, j] .= 0
                A12[k, :] .= 0

                j = m # move to next leaf node in series
            end

        end

        leaf_nodes = [node[2] for node in findall(x -> x == 1, sum(abs.(A12), dims=1))]

    end

    F = sort(F)
    T = sort(T)
    deleteat!(P, F)
    deleteat!(V, T)


    ### Find and eliminate trivial loops ###
    # visited = []
    # con_nodes = ∩([node[2] for node in findall(x -> x == 2, sum(abs.(A), dims=1))], findall(x -> x == 0, maximum(dtilde[V, :], dims=2)))
    # LoopBool = 0
    # while !isempty(con_nodes)
    #     j = con_nodes[1]
    #     boolean = 0
    #     temp = findall(x -> x != 0, A[:, j])
    #     l0 = temp[1]
    #     i0= findall(x -> x != 0, A[l0, :])
    #     i0 = filter(x -> x != j, i0)[1][2] 
    #     lN = temp[2]
    #     iNp1= findall(x -> x != 0, A[lN, :])
    #     iNp1 = filter(x -> x != j, lN)[1] 
    #     ij = j
    #     Ij = []
    #     while any(x -> x == i0, con_nodes)
    #         temp = findall(x -> x != 0, A[:, i0])
    #         lj = hcat(l0, lj)
    #         I0 = filter(x -> x != l0, temp)[1][2] 
    #         ij = hcat(i0, ij)
    #         i0 = findall(x -> x != 0, A[l0, :])
    #         i0 = filter(x -> x != i0 || x != ij, i0)[1][2] 
    #     end
    #     while any(x -> x == iNp1, con_nodes)
    #         temp = findall(x -> x != 0, A[:, iNp1])
    #         lj = hcat(lj, lN)
    #         lN = filter(x -> x != lN, temp)[1] 
    #         ij = hcat(ij, iNp1)
    #         iNp1= findall(x -> x != 0, A[lN, :])
    #         iNp1 = filter(x -> x != iNp1 || x != ij, iNp1)[1][2]
    #     end
    #     if i0 <= length(V) && iNp1 <= length(V)
    #         if V[i0] == V[iNp1]
    #             LoopBool = 1
    #             bool = 1
    #             LinkList = hcat(l0, lj, lN)
    #             F = hcat(F, P[l0], P[lj], P[lN])
    #             QSol(hcat(P[l0], P[lj], P[lN]), :) .= 0
    #             temp = findall(x -> x != 0, A[:, i0])
    #             findall(x -> x == temp, hcat(l0, lj, iN))
    #             otherlinks = filter(x -> x != (findall(x -> x == temp, hcat(l0, lj, iN))), temp)
    #             if length(otherlinks) == 1
    #                 # now this is a branch
    #                 k = otherlinks
    #                 mtemp = findall(x -> x != 0, A[k, :])
    #                 m = filter(x -> x != i0, mtemp)[1][2]
    #                 if m <= length(V)
    #                     QSol[P[k], :] = A[k, i0] .* dtilde[V[i0], :]
    #                     dtilde[V[m], :] = dtilde[V[m], :] + dtilde[V[i0], :]
    #                 else
    #                     QSol[P[k], :] = A[k, i0] .* dtilde[V[i0], :]
    #                 end
                    
    #                 push!(T, hcat(V[i0], V[ij]))
    #                 push!(F, P[k])
    #                 deleteat!(V, hcat(ij, i0))
    #                 deleteat!(P, hcat(l0, lj, lN, k))
    #                 A_temp = A[P, V]
                    
    #             elseif length(otherlinks) > 1
    #                 # do nothing
    #                 push!(T, V[ij])
    #                 deleteat!(V, ij)
    #                 deleteat!(P, hcat(l0, lj, lN))
    #                 A_temp = A[P, V]
    #             end
    #         end
    #     end
    #     if bool
    #         con_nodes = ∩([node[2] for node in findall(x -> x == 2, sum(abs.(A_temp), dims=1))], findall(x -> x == 0, maximum(dtilde[V, :], dims=2)))
    #         con_nodes = setdiff(con_nodes, findall(x -> x == V, visted))
    #     else
    #         visited = hcat(visited, V[ij])
    #         con_nodes = ∩([node[2] for node in findall(x -> x == 2, sum(abs.(A_temp), dims=1))], findall(x -> x == 0, maximum(dtilde[V, :], dims=2)))
    #         con_nodes = setdiff(con_nodes, findall(x -> x == V, visted))
    #     end
    # end


    ### Find core-series links ###
    con_nodes = V[[node[2] for node in findall(x -> x == 2, sum(abs.(A[P, V]), dims=1))]]

    for nodeIdx in con_nodes
        linkSeq = findall(x -> x != 0, A[P, nodeIdx])
        if any(x -> x == P[linkSeq[1]], S) && any(x -> x == P[linkSeq[2]], S)
            # do nothing
        elseif !any(x -> x == P[linkSeq[1]], S) && any(x -> x == P[linkSeq[2]], S)
            l = linkSeq[1]
            m = linkSeq[2]
            um = spzeros(np)
            um[P[m]] = 1
            E[P[l], :] = -A12[P[l], nodeIdx] * A12[P[m], nodeIdx] * um
            Econst[P[l], :] = A12[P[l], nodeIdx] * dtilde[nodeIdx, :]
            push!(S, P[l])
        elseif !any(x -> x == P[linkSeq[1]], S) && !any(x -> x == P[linkSeq[2]], S)
            l = linkSeq[1]
            m = linkSeq[2]
            um = spzeros(np)
            um[P[m]] = 1
            E[P[l], :] = -A12[P[l], nodeIdx] * A12[P[m], nodeIdx] * um
            Econst[P[l], :] = A12[P[l], nodeIdx] * dtilde[nodeIdx, :]
            push!(S, P[l])
        elseif any(x -> x == P[linkSeq[1]], S) && !any(x -> x == P[linkSeq[2]], S)
            l = linkSeq[2]
            m = linkSeq[1]
            um = spzeros(np)
            um[P[m]] = 1
            E[P[l], :] = -A12[P[l], nodeIdx] * A12[P[m], nodeIdx] * um
            Econst[P[l], :] = A12[P[l], nodeIdx] * dtilde[nodeIdx, :]
            push!(S, P[l])
        end
        
    end
    
    J = S[vec(maximum(abs.(E[S, S]), dims=1) .> 0)]
    while !isempty(J)
        jj = 1
        ej = E[J[jj], :]
        ji = findall(x -> x != 0, E[:, J[jj]])[1]
        uj = zeros(size(E, 2))
        uj[J[jj]] = 1
        Econst[ji, :] = Econst[ji, :] + E[ji, J[jj]] * Econst[J[jj], :]
        E[ji, :] = E[ji, :] - E[ji, J[jj]] * uj + E[ji, J[jj]] * ej
        J = S[vec(maximum(abs.(E[S, S]), dims=1) .> 0)]
    end

    P = setdiff(P, S)
    P = sort(P)
    S = sort(S)
    network_decomp = (V=V, P=P, E=E, e=Econst, QF=QSol[F, :], F=F, S=S)

    return network_decomp

end