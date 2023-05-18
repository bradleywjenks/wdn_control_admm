### misc. plots for ECC23 slide deck ###

# using OpWater
# using Plots; pgfplotsx()
using PGFPlotsX
using FileIO
using JLD2
using Colors
using ColorSchemes
using LaTeXStrings

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsmath}")

colors = ColorSchemes.Set1_9

# load network data
@load("data/network/bwfl_2022_05_hw_network.jld2")

# extract demand data
demands_agg = vec(sum(network.d, dims=1))
append!(demands_agg, demands_agg[1])

### system demand plot ### 
# quick plot
begin
    x = collect(0:0.25:24)
    plot(x, demands_agg)
end

# PGFPlotsX
begin
    x = collect(0:0.25:24)
    @pgf plot = Axis(
        {
            xlabel = "Time [h]",
            ylabel = "System demand [L/s]",
            ymin = 0,
            ymax = 80,
            xmin = 0,
            xmax = 24,
            xtick = "{0, 4, ..., 24}",
            tick_style = "black",
            width = "10cm",
            height = "6cm",
        },
        VBand(
            {
                style = "ultra thick",
                # draw = colors[1],
                draw = "none",
                fill = colors[1],
                fill_opacity = "0.1", 
                label = "SCC mode",
                # mark_options = {"solid, fill_opacity=0.15"}
                }, 
               9.25, 10.75
            ),
        VBand(
            {
                style = "ultra thick",
                # draw = colors[2],
                draw = "none",
                fill = colors[2],
                fill_opacity = "0.1", 
                label = "AZP mode",
                # mark_options = {"solid, fill_opacity=0.15"}
                }, 
                0.1, 9.1
            ),
        VBand(
            {
                style = "ultra thick",
                # draw = colors[2],
                draw = "none",
                fill = colors[2],
                fill_opacity = "0.1", 
                # mark_options = {"solid, fill_opacity=0.15"}
                }, 
                10.9, 23.9
            ),
        PlotInc(
            {
                style = "solid, very thick",
                mark = "none",
                color = "black",
            },
            Coordinates(x, demands_agg)
        ),
        [
            raw"\node ",
                {
                    draw = "none",
                    color = colors[2],
                    rotate = 90,
                },
            " at ",
            Coordinate(0.85, 8.75),
            raw"{\textbf{AZP}};"
        ], 
        [
            raw"\node ",
                {
                    draw = "none",
                    color = colors[1],
                    rotate = 90,
                },
            " at ",
            Coordinate(10, 8.75),
            raw"{\textbf{SCC}};"
        ],
        [
            raw"\node ",
                {
                    draw = "none",
                    color = colors[2],
                    rotate = 90,
                },
            " at ",
            Coordinate(23.1, 8.75),
            raw"{\textbf{AZP}};"
        ],
    )

    pgfsave("plots/control_horizon.pdf", plot)
    pgfsave("plots/control_horizon.svg", plot)
    pgfsave("plots/control_horizon.tex", plot; include_preamble=false)
    plot

end


### PCV profile plot example ###
begin
    net_name = "bwfl_2022_05_hw"
    n_v = 3
    n_f = 4
    @load "data/single_objective_results/"*net_name*"_azp_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2" sol_best
    v_loc = sol_best.v
    η = sol_best.η
    @load("data/bwfl_pcv_controls_ECC23.jld2")

    x = collect(0:0.25:24)
    η_ex = hcat(η[v_loc, :], repeat([Inf], length(v_loc)))

    @pgf η_plot = Axis(
        {
            xlabel = "Time [h]",
            ylabel = "PCV setting [m]",
            # ymin = 0,
            # ymax = 80,
            # ymajorticks = false,
            xmin = 0,
            xmax = 24,
            xtick = "{0, 4, ..., 24}",
            tick_style = "black",
            # legend_pos = "outer north east",
            width = "10cm",
            height = "8cm",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
        },
        PlotInc(
            {
                style = "solid, very thick",
                mark = "none",
                color = colors[2],
            },
            Coordinates(x, η_ex[1, :].*-1)
        ),
        LegendEntry("PCV-1"),
        PlotInc(
            {
                style = "dashed, very thick",
                mark = "none",
                color =  colors[2],
            },
            Coordinates(x, η_ex[2, :].*-1)
        ),
        LegendEntry("PCV-2"),
        PlotInc(
            {
                style = "dotted, very thick",
                mark = "none",
                color =  colors[2],
            },
            Coordinates(x, η_ex[3, :].*-1)
        ),
        LegendEntry("PCV-3"),
    )

    pgfsave("plots/pcv_settings.pdf", η_plot)
    pgfsave("plots/pcv_settings.svg", η_plot)
    pgfsave("plots/pcv_settings.tex", η_plot; include_preamble=false)
    η_plot

end


### AFV profile plot example ###
begin
    net_name = "bwfl_2022_05_hw"
    n_v = 3
    n_f = 4
    @load "data/single_objective_results/"*net_name*"_scc_nv_"*string(n_v)*"_nf_"*string(n_f)*".jld2" sol_best
    y_loc = sol_best.y
    @load("data/bwfl_afv_controls_ECC23.jld2")

    x = collect(0:0.25:24)
    α_ex = hcat(α[y_loc, :], repeat([Inf], length(y_loc)))

    @pgf α_plot = Axis(
        {
            xlabel = "Time [h]",
            ylabel = "AFV setting [L/s]",
            # ymin = 0,
            # ymax = 80,
            # ymajorticks = false,
            xmin = 0,
            xmax = 24,
            xtick = "{0, 4, ..., 24}",
            tick_style = "black",
            # legend_pos = "outer north east",
            width = "10cm",
            height = "8cm",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
        },
        PlotInc(
            {
                style = "solid, very thick",
                mark = "none",
                color = colors[1],
            },
            Coordinates(x, α_ex[1, :])
        ),
        LegendEntry("AFV-1"),
        PlotInc(
            {
                style = "dashed, very thick",
                mark = "none",
                color =  colors[1],
            },
            Coordinates(x, α_ex[2, :])
        ),
        LegendEntry("AFV-2"),
        PlotInc(
            {
                style = "dotted, very thick",
                mark = "none",
                color =  colors[1],
            },
            Coordinates(x, α_ex[3, :])
        ),
        LegendEntry("AFV-3"),
    )

    pgfsave("plots/afv_settings.pdf",α_plot)
    pgfsave("plots/afv_settings.svg", α_plot)
    pgfsave("plots/afv_settings.tex", α_plot; include_preamble=false)
    α_plot

end


### SCC objective function plot ### 
begin
    # plotting parameters
    u_vec = collect(-0.4:0.005:0.4)
    umin = 0.2
    ρ = 50

    # scc sigmoid function values
    f_σ = zeros(length(u_vec))
    for (i, u) ∈ enumerate(u_vec)
        f_σ[i] = 1 ./ (1 .+ exp(-ρ .* (u .- umin))) .+ 1 ./ (1 .+ exp(-ρ.*((-1 .* u) .- umin)))
    end

    # indicator function
    f_κx = [u_vec[1],-umin, -umin, umin, umin, u_vec[end]]
    f_κy = [1, 1, 0, 0, 1, 1]

    # PGFplotsX code
    @pgf scc_plot = Axis(
        {
            xlabel = L"$q_{j, t}$",
            ylabel = L"$\psi$",
            # ymin = 0,
            # ymax = 80,
            ymajorticks = false,
            axis_lines = "middle",
            # xmin = 0,
            # xmax = 24,
            # xtick = "{0, 4, ..., 24}",
            xmajorticks = false,
            tick_style = "black",
            legend_pos = "north east",
            # width = "10cm",
            # height = "8cm",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
            # axis_equal = true,
            # ylabel_style="{anchor=east,
            #     rotate = 90,
            #     at={(axis cs:-0.1,0.75)},
            #     font=\\Large,
            #   }",
            # style = "{origin}",
        },
        PlotInc(
            {
                style = "dashed, very thick",
                mark = "none",
                color = "black",
            },
            Coordinates(f_κx, f_κy)
        ),
        LegendEntry("Indicator function"),
        PlotInc(
            {
                style = "solid, very thick",
                mark = "none",
                color =  colors[1],
            },
            Coordinates(u_vec, f_σ)
        ),
        LegendEntry("Sum of logistic functions"),
    )

    pgfsave("plots/scc_objective.pdf", scc_plot)
    pgfsave("plots/scc_objective.svg", scc_plot)
    pgfsave("plots/scc_objective.tex", scc_plot; include_preamble=false)
    scc_plot

end