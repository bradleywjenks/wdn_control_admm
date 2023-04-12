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


