using OpWater
using FileIO

### load network data ###
begin
    net_name = "bwfl_2022_05_hw"
    # net_name = "L_town"

    # load network data
    if net_name == "bwfl_2022_05_hw"
        load_name = "bwfl_2022_05/hw"
    else
        load_name = net_name
    end
    network = load_network(load_name, afv_idx=false, dbv_idx=false, pcv_idx=false, bv_open=bv_open)

    # assign control valve locations
    nv = 3
    nf = 4
    data = load("data/problem_data/"*net_name*"_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2")
    network.pcv_loc = data["v_loc"]
    network.afv_loc = data["y_loc"]
end

### plotting code ###
plot_network_layout(network, pipes=true, reservoirs=true, pcvs=true, afvs=true, legend=true)




# ### pressure heads (optimal dfc) ###
# begin
#     t = 40 # time step
#     node_key = "pressure head"
#     p = hk .- data["elev"]
#     # p = network.elev
#     node_values = vcat(p[:, t], repeat([0.0], network.n0))
#     network.pcv_loc = data["v_loc"]
#     network.afv_loc = data["y_loc"]
#     plot_network_nodes(network, node_values=node_values, node_key=node_key, clims=(0, 80))
#     plot_network_layout(network, pipes=false, reservoirs=true, pcvs=true, afvs=true, legend=false)
# end


# ### maximum pipe flow velocities (optimal dfc) ###
# begin
#     edge_key = "velocity"
#     A = 1 ./ ((π/4).*data["D"].^2)
#     v = qk ./ 1000 .* A
#     edge_values = v[:, t]
#     network.pcv_loc = data["v_loc"]
#     network.afv_loc = data["y_loc"]
#     plot_network_edges(network, edge_values=edge_values, edge_key=edge_key, clims=(0, 0.4))
#     plot_network_layout(network, pipes=false, reservoirs=true, pcvs=true, afvs=true, legend=false)
# end


