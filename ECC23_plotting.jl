### misc. plots for ECC23 slide deck ###

using OpWater
# using Plots; pgfplotsx()
using PGFPlotsX
using FileIO
using JLD2
using Colors
using ColorSchemes
using LaTeXStrings

colors = ColorSchemes.Set1_9

# load network data
@load("data/network/bwfl_2022_05_hw_network.jld2")

# extract demand data
demands_agg = vec(sum(network.d, dims=1))
append!(demands_agg, demands_agg[1])

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