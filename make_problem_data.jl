### make problem data for ADMM script ###
using OpWater
using FileIO
using JLD2
include("src/admm_functions.jl")

# set problem parameters
begin
    # net_name = "bwfl_2022_05_hw"
    net_name = "modena"

    bv_open = false # for bwfl network only

    # scc_time = collect(38:42) # bwfl (peak) and L_town
    scc_time = collect(7:8) # Modena

    n_v = 3
    n_f = 4
    αmax = 25
    umin = 0.2
    pmin = 15
    ρ = 50

end


# make network and problem data
begin

    # load network data
    if net_name == "bwfl_2022_05_hw"
        load_name = "bwfl_2022_05/hw"
    else
        load_name = net_name
    end
    network = load_network(load_name, afv_idx=false, dbv_idx=false, pcv_idx=false, bv_open=bv_open)

    # make optimization parameters
    opt_params = make_prob_data(network; αmax_mul=αmax, umin=umin, ρ=ρ, pmin=pmin)
    q_init, h_init, err, iter = hydraulic_simulation(network, opt_params)
    S = (π * (network.D) .^ 2) / 4
    v0 = q_init ./ (1000 * S) 
    max_v = ceil(maximum(abs.(v0)))
    opt_params.Qmin , opt_params.Qmax, opt_params.umax = q_bounds_from_u(network, q_init; max_v=max_v)

    # load pcv and afv locations
    @load "data/single_objective_results/"*net_name*"_azp_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2" sol_best
    v_loc = sol_best.v
    v_dir = Int.(sign.(sol_best.q[v_loc, 1]))
    @load "data/single_objective_results/"*net_name*"_scc_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2" sol_best
    y_loc = sol_best.y

    # save problem data
    make_object_data(net_name, network, opt_params, v_loc, v_dir, y_loc, scc_time)
    
end