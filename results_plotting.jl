"""

    PLEASE NOTE:
    - This script comprises miscellaneous PGFPlotsX code for plotting ADMM results
    - It is very messy and is not intended to be reproduced
    - Use at your own risk

"""



using PGFPlotsX
using Colors
using ColorSchemes
using LaTeXStrings
using FileIO
using GraphPlot
using Graphs
using Statistics

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}")

### define colours ###
begin
    cRed = colorant"#c1272d"
    cNavy = colorant"#0000a7"
    cTeal = colorant"#008176"
    cYellow = colorant"#eecc16"
    # cGrey = colorant"#595959"
    # n = 3
    # gg_colors = [LCHuv(65, 100, h) for h in range(15, 360+15, n+1)][1:n] # ggplot2 colors
    # colors = ColorSchemes.seaborn_colorblind
    # colors = ColorSchemes.tableau_colorblind
    # colors = [colorant"#000000", colorant"#E69F00", colorant"#56B4E9", colorant"#009E73", colorant"#F0E442", colorant"#0072B2", colorant"#D55E00", colorant"#CC79A7"]
    reds = ColorSchemes.Reds
    blues = ColorSchemes.Blues
    greens = ColorSchemes.Greens
    purples = ColorSchemes.Purples
    greys = ColorSchemes.Greys
    cGrey = greys[6]
    # colors = ColorSchemes.Reds_5[2:4]
    # paired = ColorSchemes.Paired_10
    colors = ColorSchemes.Set1_9
    # colors = ColorSchemes.Egypt
    # colors = ColorSchemes.Dark2_8
    # colors = [colorant"#ED6E00", colorant"#0000FF", colorant"#00AAAA"]
    # colors = [colorant"#2E8B57", colorant"#572E8B", colorant"#8B572E"]
    # colors = ColorSchemes.seaborn_bright
end


### problem parameters ###
begin
    # net_name = "L_town"
    # nt = 96
    # np = 797
    # nn = 688
    # δ_1 = [25, 15, 5]
    # δ_2 = [30, 20, 10]
    # δ_3 = [15, 10, 5]
    # γ_hat = 0.1

    net_name = "bwfl_2022_05_hw"
    nt = 96
    np = 2816
    nn = 2745
    δ_1 = [25, 20, 15]
    δ_2 = [20, 15, 10]
    δ_3 = [15, 10, 5]
    γ_hat = 0.1

    scc_time = collect(38:42) 
    kmax = 1000
end


### load results data ###
begin
    γ_range = 10.0 .^ (-3:1:2)
    δ_p1 = Array{Union{Any}}(nothing, length(γ_range), 4, 3); δ_p1[:, 1, :] .= γ_range
    δ_p2 = Array{Union{Any}}(nothing, length(γ_range), 4, 3); δ_p2[:, 1, :] .= γ_range
    δ_p3 = Array{Union{Any}}(nothing, length(γ_range), 4, 3); δ_p3[:, 1, :] .= γ_range
    residuals_p1 = Array{Union{Any}}(nothing, length(γ_range), kmax, 2, 3) # D1 = γ values; D2 = residual data; D3 = primal or dual residual data; D3 = δ value
    residuals_p2 = Array{Union{Any}}(nothing, length(γ_range), kmax, 2, 3) # D1 = γ values; D2 = residual data; D3 = primal or dual residual data; D3 = δ value
    residuals_p3 = Array{Union{Any}}(nothing, length(γ_range), kmax, 2, 3) # D1 = γ values; D2 = residual data; D3 = primal or dual residual data; D3 = δ value
    f_azp = Array{Union{Any}}(nothing, 4, nt, 3)
    f_scc = Array{Union{Any}}(nothing, 4, nt, 3)
    xk_0 = Array{Union{Any}}(nothing, 2*np+2*nn, nt, 3)
    xk_1 = Array{Union{Any}}(nothing, 2*np+2*nn, nt, 3)
    xk_2 = Array{Union{Any}}(nothing, 2*np+2*nn, nt, 3)
    xk_3 = Array{Union{Any}}(nothing, 2*np+2*nn, nt, 3)

    for (i, v) ∈ enumerate(γ_range)
        if v ≥ 1
            v = Int(v)
        else
            v = round(v, digits=3)
        end 
        
        # load and organise data
        for n ∈ collect(1:length(δ_1))

            # pressure variability constraint (p1)
            temp_p1 = load("data/admm_results/"*net_name*"_variability_delta_"*string(δ_1[n])*"_gamma_"*string(v)*"_distributed.jld2")
            δ_p1[i, 2, n] = sum(temp_p1["f_val"])
            if length(temp_p1["p_residual"]) == 1000
                δ_p1[i, 3, n] = Inf
            else
                δ_p1[i, 3, n] = length(temp_p1["p_residual"])
            end
            δ_p1[i, 4, n] = temp_p1["cpu_time"]
            residuals_p1[i, 1:length(temp_p1["p_residual"]), 1, n] .= temp_p1["p_residual"]
            residuals_p1[i, 1:length(temp_p1["d_residual"]), 2, n] .= temp_p1["d_residual"]
            if v == γ_hat
                if n == 1
                    f_azp[1, :, 1] = temp_p1["f_azp"]
                    f_scc[1, :, 1] = temp_p1["f_scc"]
                    xk_0[:, :, 1] .= temp_p1["xk_0"]
                    xk_1[:, :, 1] .= temp_p1["xk"]
                elseif n == 2
                    xk_2[:, :, 1] .= temp_p1["xk"]
                elseif n == 3
                    xk_3[:, :, 1] .= temp_p1["xk"]
                end
                f_azp[n+1, :, 1] = temp_p1["f_azp_pv"]
                f_scc[n+1, :, 1] = temp_p1["f_scc_pv"]
            end

            # pressure range constraint (p2)
            temp_p2 = load("data/admm_results/"*net_name*"_range_delta_"*string(δ_2[n])*"_gamma_"*string(v)*"_distributed.jld2")
            δ_p2[i, 2, n] = sum(temp_p2["f_val"])
            if length(temp_p2["p_residual"]) == 1000
                δ_p2[i, 3, n] = Inf
            else
                δ_p2[i, 3, n] = length(temp_p2["p_residual"])
            end
            δ_p2[i, 4, n] = temp_p2["cpu_time"]
            residuals_p2[i, 1:length(temp_p2["p_residual"]), 1, n] = temp_p2["p_residual"]
            residuals_p2[i, 1:length(temp_p2["d_residual"]), 2, n] = temp_p2["d_residual"]
            if v == γ_hat
                if n == 1
                    f_azp[1, :, 2] = temp_p2["f_azp"]
                    f_scc[1, :, 2] = temp_p2["f_scc"]
                    xk_0[:, :, 2] .= temp_p2["xk_0"]
                    xk_1[:, :, 2] .= temp_p2["xk"]
                elseif n == 2
                    xk_2[:, :, 2] .= temp_p2["xk"]
                elseif n == 3
                    xk_3[:, :, 2] .= temp_p2["xk"]
                end
                f_azp[n+1, :, 2] = temp_p2["f_azp_pv"]
                f_scc[n+1, :, 2] = temp_p2["f_scc_pv"]
            end

            # pressure variation constraint (p3)
            temp_p3 = load("data/admm_results/"*net_name*"_variation_delta_"*string(δ_3[n])*"_gamma_"*string(v)*"_distributed.jld2")
            δ_p3[i, 2, n] = sum(temp_p3["f_val"])
            if length(temp_p3["p_residual"]) == 250
                δ_p3[i, 3, n] = Inf
            else
                δ_p3[i, 3, n] = length(temp_p3["p_residual"])
            end
            δ_p3[i, 4, n] = temp_p3["cpu_time"]
            residuals_p3[i, 1:length(temp_p3["p_residual"]), 1, n] = temp_p3["p_residual"]
            residuals_p3[i, 1:length(temp_p3["d_residual"]), 2, n] = temp_p3["d_residual"]
            if v == γ_hat
                if n == 1
                    f_azp[1, :, 3] = temp_p3["f_azp"]
                    f_scc[1, :, 3] = temp_p3["f_scc"]
                    xk_0[:, :, 3] .= temp_p3["xk_0"]
                    xk_1[:, :, 3] .= temp_p3["xk"]
                elseif n == 2
                    xk_2[:, :, 3] .= temp_p3["xk"]
                elseif n == 3
                    xk_3[:, :, 3] .= temp_p3["xk"]
                end
                f_azp[n+1, :, 3] = temp_p3["f_azp_pv"]
                f_scc[n+1, :, 3] = temp_p3["f_scc_pv"]
            end

        end
    end
    f_azp_0 = f_azp[1, :, 1]
    f_scc_0 = f_scc[1, :, 1]
end

### objective value plot ###
begin
        # pv_type = "variability"
        pv_type = "range"
        # pv_type = "variation"
    
        # organise values
        if pv_type == "variability"
            δ_p = δ_p1
            c = colors[3]
            # ymin_obj = 3220
            # ymax_obj = 3290
            # ytick = "{3220, 3230, ..., 3290}"
            # ymax_iter = 200
            ymin_obj = 3220
            ymax_obj = 3270
            ytick_obj = "{3220, 3230, ..., 3270}"
            ymax_iter = 60
            ytick_iter = "{0, 15, ..., 60}"
            name = L"$(\mathcal{C}_v)$"
        elseif pv_type == "range"
            δ_p = δ_p2
            # c = colors[2]
            c = colors[4]
            ymin_obj = 3200
            ymax_obj = 3360
            ytick_obj = "{3200, 3240, ..., 3360}"
            ymax_iter = 60
            ytick_iter = "{0, 15, ..., 60}"
            # ymin_obj = 2850
            # ymax_obj = 3100
            # ytick_obj = "{2850, 2900, ..., 3100}"
            # ymax_iter = 250
            name = L"(\mathcal{C}_r)"
        elseif pv_type == "variation"
            δ_p = δ_p3
            # c = cGrey
            c = colors[3]
            ymin_obj = 3220
            ymax_obj = 3280
            ytick_obj = "{3220, 3230, ..., 3280}"
            ymax_iter = 400
            ytick_iter = "{0, 80, ..., 400}"
            name = L"$(\mathcal{C}_v)$"
        end

    @pgf obj_plot = Axis(
        {
            xmode = "log",
            xlabel = L"\gamma",  # L"()" shows $()$ in latex math
            ylabel = "Objective value",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
            scaled_y_ticks = "{base 10:-3}",
            ymin = ymin_obj,
            ymax = ymax_obj,
            ytick = ytick_obj,
            tick_style = "black",
            y_tick_label_style = "{/pgf/number format/fixed zerofill}",
            # ymin = 3220,
            # ymax = 3290,
            # ytick = "{3220, 3230, ..., 3290}",
            # minor_y_tick_num = 10,
            # legend_pos = "outer north east",
        },
        PlotInc(
            {
                style = "solid, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = {"solid, fill_opacity=0.15"},
                color = c,
            },
            Coordinates(γ_range, δ_p[:, 2, 1])
        ), 
        # LegendEntry(L"\delta_1 \,"*name),
        PlotInc(
            {
                style = "dashed, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = {"solid, fill_opacity=0.15"},
                color = c,
            },
            Coordinates(γ_range, δ_p[:, 2, 2])
        ), 
        # LegendEntry(L"\delta_2 \,"*name),
        PlotInc(
            {
                style = "dotted, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = {"solid, fill_opacity=0.15"},
                color = c,
            },
            Coordinates(γ_range, δ_p[:, 2, 3])
        ), 
        # LegendEntry(L"\delta_3 \,"*name),
        HLine(
            { 
                style = "solid, very thick", 
                color = "black", 
                # color = colors[9],
                }, 
                sum(setdiff(f_azp_0, f_azp_0[scc_time])) + sum(f_scc_0[scc_time])*-1
            ),
            # LegendEntry(L"$\delta_{\infty}$"),
    )

    ### no. of iterations plot ###
    @pgf iter_plot = Axis(
        {
            xmode = "log",
            # ymode = "log",
            xlabel = L"\gamma",  # L"()" shows $()$ in latex math
            ylabel = "Iterations",
            ymin = 0,
            ymax = ymax_iter,
            tick_style = "black",
            ytick = ytick_iter,
            legend_pos = "north east",
            # legend_pos = "outer north east",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
        },
        PlotInc(
            {
                style = "solid, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = {"solid, fill_opacity=0.15"},
                color = c,
            },
            Coordinates(γ_range, δ_p[:, 3, 1])
        ), 
        LegendEntry(L"\delta_1 \,"*name),
        PlotInc(
            {
                style = "dashed, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = {"solid, fill_opacity=0.15"},
                color = c,
            },
            Coordinates(γ_range, δ_p[:, 3, 2])
        ), 
        LegendEntry(L"\delta_2 \,"*name),
        PlotInc(
            {
                style = "dotted, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = {"solid, fill_opacity=0.15"},
                color = c,
            },
            Coordinates(γ_range, δ_p[:, 3, 3])
        ), 
        LegendEntry(L"\delta_3 \,"*name),
        PlotInc(
            {
                style = "solid, very thick",
                mark = "none",
                color = "black", 
                # color = colors[9],
            },
            Coordinates(γ_range, repeat([-1], length(γ_range)))
        ), 
            LegendEntry(L"$\delta_{\infty}$"),
    )

    ### combine obj and iter plots ###
    obj_iter_plot = @pgf GroupPlot(
        { 
            group_style = {
                group_size = "2 by 1",
                # xticklabels_at = "edge bottom",
                # yticklabels_at = "edge left",
                horizontal_sep = "1.6cm",
                },
        },
    obj_plot, iter_plot)
    pgfsave("plots/"*net_name*"_"*pv_type*"_obj_iter.pdf", obj_iter_plot)
    pgfsave("plots/"*net_name*"_"*pv_type*"_obj_iter.svg", obj_iter_plot)
    pgfsave("plots/"*net_name*"_"*pv_type*"_obj_iter.tex", obj_iter_plot; include_preamble=false)
    obj_iter_plot
end


### pressure cdf plots ###
begin
    pv_type = "variability"
    # pv_type = "range"
    # pv_type = "variation"

    # compute values
    if pv_type == "variability"
        hk_0 = xk_0[np+1:np+nn, :, 1]
        hk_1 = xk_1[np+1:np+nn, :, 1]
        hk_2 = xk_2[np+1:np+nn, :, 1]
        hk_3 = xk_3[np+1:np+nn, :, 1]
    elseif pv_type == "range"
        hk_0 = xk_0[np+1:np+nn, :, 2]
        hk_1 = xk_1[np+1:np+nn, :, 2]
        hk_2 = xk_2[np+1:np+nn, :, 2]
        hk_3 = xk_3[np+1:np+nn, :, 2]
    elseif pv_type == "variation"
        hk_0 = xk_0[np+1:np+nn, :, 3]
        hk_1 = xk_1[np+1:np+nn, :, 3]
        hk_2 = xk_2[np+1:np+nn, :, 3]
        hk_3 = xk_3[np+1:np+nn, :, 3]
    end

    pv_0 = zeros(nn)
    pv_1 = zeros(nn)
    pv_2 = zeros(nn)
    pv_3 = zeros(nn)

    if pv_type == "variation"
        pv_0 = [maximum([abs(hk_0[i, k] - hk_0[i, k-1]) for k ∈ collect(2:nt)]) for i ∈ collect(1:nn)]
        pv_1 = [maximum([abs(hk_1[i, k] - hk_1[i, k-1]) for k ∈ collect(2:nt)]) for i ∈ collect(1:nn)]
        pv_2 = [maximum([abs(hk_2[i, k] - hk_2[i, k-1]) for k ∈ collect(2:nt)]) for i ∈ collect(1:nn)]
        pv_3 = [maximum([abs(hk_3[i, k] - hk_3[i, k-1]) for k ∈ collect(2:nt)]) for i ∈ collect(1:nn)]
    elseif pv_type == "variability"
        pv_0 = [sqrt(sum((hk_0[i, k] - hk_0[i, k-1])^2 for k ∈ collect(2:nt))) for i ∈ collect(1:nn)]
        pv_1 = [sqrt(sum((hk_1[i, k] - hk_1[i, k-1])^2 for k ∈ collect(2:nt))) for i ∈ collect(1:nn)]
        pv_2 = [sqrt(sum((hk_2[i, k] - hk_2[i, k-1])^2 for k ∈ collect(2:nt))) for i ∈ collect(1:nn)]
        pv_3 = [sqrt(sum((hk_3[i, k] - hk_3[i, k-1])^2 for k ∈ collect(2:nt))) for i ∈ collect(1:nn)]
    elseif pv_type == "range"
        pv_0 = [maximum(hk_0[i, :]) - minimum(hk_0[i, :]) for i ∈ collect(1:nn)]
        pv_1 = [maximum(hk_1[i, :]) - minimum(hk_1[i, :]) for i ∈ collect(1:nn)]
        pv_2 = [maximum(hk_2[i, :]) - minimum(hk_2[i, :]) for i ∈ collect(1:nn)]
        pv_3 = [maximum(hk_3[i, :]) - minimum(hk_3[i, :]) for i ∈ collect(1:nn)]
    end

    pv_0_cdf = sort(vec(pv_0))
    pv_1_cdf = sort(vec(pv_1))
    pv_2_cdf = sort(vec(pv_2))
    pv_3_cdf = sort(vec(pv_3))
    y = collect(1:nn)./(nn)

    # define xlabel and x bounds
    if pv_type == "variation"
        xlabel = "Pressure variation [m]" 
        xmin = 0
        xmax = 25
        # c = cGrey
        c = colors[3]
    elseif pv_type == "variability"
        xlabel = "Pressure variability [m]"
        # xmin = 0
        # xmax = 60
        xmin = 5
        xmax = 35
        c = colors[3]
    elseif pv_type == "range"
        xlabel = "Pressure range [m]" 
        xmin = 0
        xmax = 30
        # xmin = 0
        # xmax = 60
        # c = colors[2]
        c = colors[4]
    end

    # generate plot
    @pgf cdf_plot = Axis(
        {
            # xmajorgrids, # show grids along x axis
            # ymajorgrids, # show grids along y axis
            ylabel = "Cumulative probability",
            xlabel = xlabel,
            xmin = xmin,
            xmax = xmax,
            ymin = 0,
            ymax = 1,
            tick_style = "black",
            legend_pos = "south east",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
        },
    PlotInc(
        {
            style = "solid, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = colors[2],
        },
        Coordinates(pv_1_cdf, y)
    ), 
    LegendEntry(L"$\delta_1$"),
    PlotInc(
        {
            style = "dashed, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = colors[2],
        },
        Coordinates(pv_2_cdf, y)
    ), 
    LegendEntry(L"$\delta_2$"),
    PlotInc(
        {
            style = "dotted, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = colors[2],
        },
        Coordinates(pv_3_cdf, y)
    ), 
    LegendEntry(L"$\delta_3$"),
    PlotInc(
        {
            style = "solid, very thick",
            mark = "none",
            color = "black",
            # color = sea[1],
            # color = colors[8],
            # color = cGrey,
        },
        Coordinates(pv_0_cdf, y)
    ), 
    LegendEntry(L"$\delta_{\infty}$")
    )
pgfsave("plots/"*net_name*"_"*pv_type*"_cdf.pdf", cdf_plot)
pgfsave("plots/"*net_name*"_"*pv_type*"_cdf.svg", cdf_plot)
pgfsave("plots/"*net_name*"_"*pv_type*"_cdf.tex", cdf_plot; include_preamble=false)
cdf_plot
end



### objective time series plot ###
begin
    x = collect(0:0.25:24)

    # pv_type = "variability"
    pv_type = "range"
    # pv_type = "variation"

    # compute values
    if pv_type == "variability"
        f_1 = hcat(f_azp[:, :, 1], repeat([Inf], size(f_azp, 1)))
        f_2 = hcat(f_scc[:, :, 1], repeat([Inf], size(f_scc, 1)))
        c = colors[3]
    elseif pv_type == "range"
        f_1 = hcat(f_azp[:, :, 2], repeat([Inf], size(f_azp, 1)))
        f_2 = hcat(f_scc[:, :, 2], repeat([Inf], size(f_scc, 1)))
        # c = colors[2]
        c = colors[4]
    elseif pv_type == "variation"
        f_1 = hcat(f_azp[:, :, 3], repeat([Inf], size(f_azp, 1)))
        f_2 = hcat(f_scc[:, :, 3], repeat([Inf], size(f_scc, 1)))
        # c = cGrey
        c = colors[3]
    end

    # AZP objective plot
    azp_plot = @pgf Axis(
        {
            ylabel = "AZP [m]",
            # xlabel = {none},
            xmin = 0,
            xmax = 24,
            xtick = "{0, 4, ..., 24}",
            # ymin = 30,
            # ymax = 60,
            # ytick = "{30, 36, ..., 60}",
            ymin = 32,
            ymax = 48,
            ytick = "{32, 36, ..., 48}",
            tick_style = "black",
            legend_pos = "north east",
            scale_only_axis = true,
            width = "10cm",
            height = "3.5cm",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
        },
    PlotInc(
        {
            style = "solid, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = cRed,
        },
        Coordinates(x, f_1[2, :])
    ), 
    LegendEntry(L"$\delta_1$"),
    PlotInc(
        {
            style = "dashed, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = cTeal,
        },
        Coordinates(x, f_1[3, :])
    ), 
    LegendEntry(L"$\delta_2$"),
    PlotInc(
        {
            style = "dotted, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = cGrey,
        },
        Coordinates(x, f_1[4, :])
    ), 
    LegendEntry(L"$\delta_3$"),
    PlotInc(
        {
            style = "solid, thick",
            mark = "none",
            color = "black",
            # color = sea[1],
            # color = colors[9],
            # color = cGrey,
        },
        Coordinates(x, f_1[1, :])
    ), 
    LegendEntry(L"$\delta_{\infty}$"),
    VBand(
        {
            draw = {none},
            fill = cGrey,
            opacity = "0.2", 
            # mark_options = {"solid, fill_opacity=0.15"}
            }, 
            (scc_time[1]-1)/4, (scc_time[end]-1)/4
        ),
        LegendEntry("SCC period"),
    )


    # SCC objective plot
    scc_plot = @pgf Axis(
        {
            ylabel = L"SCC $[\%]$",
            xlabel = "Time [h]",
            xmin = 0,
            xmax = 24,
            xtick = "{0, 4, ..., 24}",
            # ymin = 10,
            # ymax = 85,
            # ytick = "{10, 25, ..., 85}",
            ymin = 5,
            ymax = 45,
            ytick = "{5, 15, ..., 45}",
            tick_style = "black",
            # legend_pos = "north east",
            scale_only_axis = true,
            width = "10cm",
            height = "3.5cm",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
        },
    PlotInc(
        {
            style = "solid, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = cRed,
        },
        Coordinates(x, f_2[2, :])
    ), 
    PlotInc(
        {
            style = "dashed, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = cTeal,
        },
        Coordinates(x, f_2[3, :])
    ), 
    PlotInc(
        {
            style = "dotted, very thick",
            mark = "none",
            # color = "black",
            # color = sea[3],
            color = c,
            # color = cGrey,
        },
        Coordinates(x, f_2[4, :])
    ), 
    PlotInc(
        {
            style = "solid, thick",
            mark = "none",
            color = "black",
            # color = sea[1],
            # color = colors[9],
            # color = cGrey,
        },
        Coordinates(x, f_2[1, :])
    ), 
    VBand(
        {
            draw = {none},
            fill = cGrey,
            opacity = "0.2", 
            # mark_options = {"solid, fill_opacity=0.15"}
            }, 
            (scc_time[1]-1)/4, (scc_time[end]-1)/4
        ),
    )

    # group AZP and SCC plots
    azp_scc_plot = @pgf GroupPlot(
        { 
            group_style = {
                group_size = "1 by 2",
                # xticklabels_at = "edge bottom",
                # yticklabels_at = "edge left",
                vertical_sep = "1.0cm",
                },
        },
    azp_plot, scc_plot)

    pgfsave("plots/"*net_name*"_"*pv_type*"_azp_scc.pdf", azp_scc_plot)
    pgfsave("plots/"*net_name*"_"*pv_type*"_azp_scc.svg", azp_scc_plot)
    pgfsave("plots/"*net_name*"_"*pv_type*"_azp_scc.tex", azp_scc_plot; include_preamble=false)
    azp_scc_plot
end



### residuals plot ###
begin

    # pv_type = "variability"
    pv_type = "range"
    # pv_type = "variation"

    # δ_idx = 1
    # δ_idx = 2
    δ_idx = 3

    # load data
    if pv_type == "variability"
        δ = δ_1
        c = greens[4:end]
        p_residual = residuals_p1[:, :, 1, δ_idx]; p_residual = map(x-> x === nothing ? Inf : x, p_residual)
        d_residual = residuals_p1[:, :, 2, δ_idx]; d_residual = map(x-> x === nothing ? Inf : x, d_residual)
        δ_name = L"\delta_{%$δ_idx}"
    elseif pv_type == "range"
        δ = δ_2
        # c = blues[4:end]
        c = purples[4:end]
        p_residual = residuals_p2[:, :, 1, δ_idx]; p_residual = map(x-> x === nothing ? Inf : x, p_residual)
        d_residual = residuals_p2[:, :, 2, δ_idx]; d_residual = map(x-> x === nothing ? Inf : x, d_residual)
        δ_name = L"\delta_{%$δ_idx}"
    elseif pv_type == "variation"
        δ = δ_3
        # c = greys[4:end]
        c = greens[4:end]
        p_residual = residuals_p3[:, :, 1, δ_idx]; p_residual = map(x-> x === nothing ? Inf : x, p_residual)
        d_residual = residuals_p3[:, :, 2, δ_idx]; d_residual = map(x-> x === nothing ? Inf : x, d_residual)
        δ_name = L"\delta_{%$δ_idx}"
    end

    if δ_idx == 1
        line_style = "solid, very thick"
    elseif δ_idx == 2
        line_style = "solid, very thick"
    elseif δ_idx == 3
        line_style = "solid, very thick"
    end

    # x-axis maximum
    num_iter = 0
    for i ∈ collect(1:size(p_residual, 1))
        if length(filter(isfinite, p_residual[i, :])) > num_iter
            num_iter = length(filter(isfinite, p_residual[i, :])) 
        end
    end
    num_iter = Int(ceil(num_iter/10)*10)

    # y-axis maximum
    ymax_p = Int(ceil(maximum(filter(isfinite, vec(p_residual)))*1.1))
    ymax_p = 400
    if ymax_p ≥ 10
        scaled_y_ticks_p = "{base 10:-2}"
        y_tick_p = "{0, 40, ..., $ymax_p}"
    else
        scaled_y_ticks_p = "false"
        # y_tick_p = "{0, 1, ..., $ymax_p}"
    end
    ymax_d = Int(ceil(maximum(filter(isfinite, vec(d_residual)))*1.1))
    if ymax_d ≥ 10
        scaled_y_ticks_d = "{base 10:-4}"
        # y_tick_d = "{0, 20, ..., $ymax_d}"
    else
        scaled_y_ticks_d = "false"
        # y_tick_d = "{0, 1, ..., $ymax_d}"
    end
        


    # p_residual plot
    p_residual_plot = @pgf Axis(
        {
            ylabel = "Residual",
            xlabel = "Iteration",
            xmin = 0,
            xmax = num_iter,
            xtick = "{0, 10, ..., $num_iter}",
            ymin = 0,
            ymax = ymax_p,
            # ytick = y_tick_p,
            tick_style = "black",
            scaled_y_ticks = scaled_y_ticks_p,
            legend_pos = "north east",
            # scale_only_axis = true,
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            # y_tick_label_style = "{/pgf/number format/fixed zerofill}",
            y_tick_label_style = "{/pgf/number format/fixed zerofill, /pgf/number format/precision=1, /pgf/number format/fixed}",
            # y_tick_label_style = "{/pgf/number format/precision=1}",
            legend_style = "{font=\\large}",
            title_style = "{font=\\Large}",
            title = "$δ_name"
        },
        # [
        #     raw"\node ",
        #         {
        #             draw = "none",
        #             color = "black",
        #             # rotate = 90,
        #             style = "{font=\\LARGE}",
        #         },
        #     " at ",
        #     Coordinate(num_iter*0.925, ymax_p*0.8),
        #     "{$δ_name};"
        # ], 
    )
    @pgf for i ∈ collect(1:size(p_residual, 1))
        γ_idx = Int(log10(γ_range[i]))
        γ_name = L"\gamma = 10^{%$γ_idx}"

        p = PlotInc(
            {
                style = line_style,
                mark = "none",
                # mark = "none",
                color = c[i],
            },
                Coordinates(collect(1:length(p_residual[i, :])), p_residual[i, :]),
        )
        # Add plot to axis
        push!(p_residual_plot, p)

        # Add legend to axis
        push!(p_residual_plot, LegendEntry("$γ_name"))
        
    end

    pgfsave("plots/"*net_name*"_"*pv_type*"_"*string(δ[δ_idx])*"_residuals.pdf", p_residual_plot)
    pgfsave("plots/"*net_name*"_"*pv_type*"_"*string(δ[δ_idx])*"_residuals.svg", p_residual_plot)
    pgfsave("plots/"*net_name*"_"*pv_type*"_"*string(δ[δ_idx])*"_residuals.tex", p_residual_plot; include_preamble=false)
    p_residual_plot



end


