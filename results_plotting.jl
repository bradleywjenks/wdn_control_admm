"""

    NOTE:
    - This script comprises miscellaneous PGFPlotsX code for plotting results
    - It is very messy and is not intended to be reproduced
    - Use at your own risk

"""



using PGFPlotsX
using Colors
using ColorSchemes
using LaTeXStrings
using FileIO
using Statistics

# push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}")

### define colours ###
begin
    cRed = colorant"#c1272d"
    cNavy = colorant"#0000a7"
    cTeal = colorant"#008176"
    cYellow = colorant"#eecc16"
    # cGrey = colorant"#595959"
    # n = 3
    # gg_colors = [LCHuv(65, 100, h) for h in range(15, 360+15, n+1)][1:n] # ggplot2 colors
    # colors = ColorSchemes.seaborn_muted
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

    # net_name = "modena"
    # nt = 24
    # np = 317
    # nn = 268
    # δ = [20, 15, 10]
    # γ_hat = 0.1

    # net_name = "L_town"
    # nt = 96
    # np = 797
    # nn = 688
    # δ = [30, 20, 10]
    # γ_hat = 0.01

    net_name = "bwfl_2022_05_hw"
    nt = 96
    np = 2816
    nn = 2745
    δ = [20, 15, 10]
    γ_hat = 0.01

    scc_time = collect(38:42) 
    # scc_time = collect(7:8)
    kmax = 1000
end


### load results data ###
begin
    γ_range = 10.0 .^ (-3:1:2)
    δ_vals = Array{Union{Any}}(nothing, length(γ_range), length(δ)+1, 3); δ_vals[:, 1, :] .= γ_range # D1 = γ values; D2 = pressure tolerances; D3 = obj. vals or no. iterations or cpu time
    residuals = Array{Union{Any}}(nothing, length(γ_range), kmax, length(δ)) # D1 = γ values; D2 = residual data; D3 = pressure tolerance value
    f_0 = Array{Union{Any}}(nothing, length(γ_range), 1, length(δ))
    f_azp = Array{Union{Any}}(nothing, length(δ)+2, nt)
    f_scc = Array{Union{Any}}(nothing, length(δ)+2, nt)
    x_k = Array{Union{Any}}(nothing, 2*np+2*nn, nt, length(δ)+2)

    for (i, v) ∈ enumerate(γ_range)
        if v ≥ 1
            v = Int(v)
        else
            v = round(v, digits=3)
        end 
        
        # load and organise data
        for n ∈ collect(1:length(δ))

            temp = load("data/two_level_results/"*net_name*"_range_"*string(δ[n])*"_beta_"*string(v)*".jld2")
            δ_vals[i, n+1, 1] = sum(temp["f_val"])
            residuals[i, :, n] .= temp["residuals"][:, 4]
            if temp["k_iter"] == 1000
                δ_vals[i, n+1, 2] = Inf # no. iterations
                δ_vals[i, n+1, 3] = Inf # cpu time
            else
                δ_vals[i, n+1, 2] = temp["k_iter"]  # no. iterations
                δ_vals[i, n+1, 3] = temp["cpu_time"] # cpu time
            end

            if v == γ_hat
                if n == 1
                    f_azp[1, :] .= temp["f_azp"]
                    f_scc[1, :] .= temp["f_scc"]
                    x_k[:, :, 1] .= temp["x_0"]
                end
                x_k[:, :, n+1] .= temp["x_k"]
                f_azp[n+1, :] = temp["f_azp_pv"]
                f_scc[n+1, :] = temp["f_scc_pv"]
            end

            f_0[i, :, n] .= sum(temp["obj_hist"][2, :])


        end

        if v == γ_hat
            temp = load("data/two_level_results/"*net_name*"_range_inf_beta_"*string(v)*".jld2")
            x_k[:, :, length(δ)+2] .= temp["x_k"]
            f_azp[length(δ)+2, :] .= temp["f_azp_pv"]
            f_scc[length(δ)+2, :] .= temp["f_scc_pv"]
        end

    end

end

### objective value plot ###
begin

            c = colors[1]
            # c = colors[2]
            # c = colors[3]

            # ymin_obj = 280
            # ymax_obj = 440
            # ytick_obj = "{280, 320, ..., 440}"
            # ymax_iter = 800
            # ytick_iter = "{0, 200, ..., 800}"

            # ymin_obj = 2850
            # ymax_obj = 3050
            # ytick_obj = "{2850, 2900, ..., 3050}"
            # ymax_iter = 350
            # ytick_iter = "{0, 70, ..., 350}"

            ymin_obj = 3220
            ymax_obj = 3320
            ytick_obj = "{3220, 3240, ..., 3320}"
            ymax_iter = 120
            ytick_iter = "{0, 30, ..., 120}"

    @pgf obj_plot = Axis(
        {
            xmode = "log",
            xlabel = L"$\beta^0$",  # L"()" shows $()$ in latex math
            ylabel = "Objective value",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
            scaled_y_ticks = "{base 10:-3}",
            # scaled_y_ticks = "{base 10:-2}",
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
                mark_options = "solid",
                color = c,
            },
            Coordinates(γ_range, δ_vals[:, 2, 1])
        ), 
        # LegendEntry(L"\delta_1 \,"*name),
        PlotInc(
            {
                style = "dashed, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = "solid",
                color = c,
            },
            Coordinates(γ_range, δ_vals[:, 3, 1])
        ), 
        # LegendEntry(L"\delta_2 \,"*name),
        PlotInc(
            {
                style = "dotted, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = "solid",
                color = c,
            },
            Coordinates(γ_range, δ_vals[:, 4, 1])
        ), 
        # LegendEntry(L"\delta_3 \,"*name),
        HLine(
            { 
                style = "solid, very thick", 
                color = "black", 
                # color = colors[9],
                }, 
                f_0[1, 1, 1]
            ),
            # LegendEntry(L"$\delta_{\infty}$"),
    )

    ### no. of iterations plot ###
    @pgf iter_plot = Axis(
        {
            xmode = "log",
            # ymode = "log",
            xlabel = L"$\beta^0$",  # L"()" shows $()$ in latex math
            ylabel = "Iterations",
            ymin = 0,
            ymax = ymax_iter,
            tick_style = "black",
            ytick = ytick_iter,
            # legend_pos = "north east",
            legend_pos = "outer north east",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            legend_style = "{font=\\large}",
        },
        PlotInc(
            {
                style = "solid, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = "solid",
                color = c,
            },
            Coordinates(γ_range, δ_vals[:, 2, 2])
        ), 
        LegendEntry(L"$\delta_1$"),
        PlotInc(
            {
                style = "dashed, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = "solid",
                color = c,
            },
            Coordinates(γ_range, δ_vals[:, 3, 2])
        ), 
        LegendEntry(L"$\delta_2$"),
        PlotInc(
            {
                style = "dotted, very thick",
                mark = "o",
                # mark_size = "3pt",
                mark_options = "solid",
                color = c,
            },
            Coordinates(γ_range, δ_vals[:, 4, 2])
        ), 
        LegendEntry(L"$\delta_3$"),
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
    pgfsave("plots/"*net_name*"_range_obj_iter.pdf", obj_iter_plot)
    pgfsave("plots/"*net_name*"_range_obj_iter.svg", obj_iter_plot)
    pgfsave("plots/"*net_name*"_range_obj_iter.tex", obj_iter_plot; include_preamble=false)
    obj_iter_plot
end


### pressure cdf plots ###
begin

    # hk_0 = x_k[np+1:np+nn, :, 1]
    hk_1 = x_k[np+1:np+nn, :, 2]
    hk_2 = x_k[np+1:np+nn, :, 3]
    hk_3 = x_k[np+1:np+nn, :, 4]
    hk_inf = x_k[np+1:np+nn, :, 5]


    # pv_0 = zeros(nn)
    pv_1 = zeros(nn)
    pv_2 = zeros(nn)
    pv_3 = zeros(nn)
    pv_inf = zeros(nn)

    # pv_0 = [maximum(hk_0[i, :]) - minimum(hk_0[i, :]) for i ∈ collect(1:nn)]
    pv_1 = [maximum(hk_1[i, :]) - minimum(hk_1[i, :]) for i ∈ collect(1:nn)]
    pv_2 = [maximum(hk_2[i, :]) - minimum(hk_2[i, :]) for i ∈ collect(1:nn)]
    pv_3 = [maximum(hk_3[i, :]) - minimum(hk_3[i, :]) for i ∈ collect(1:nn)]
    pv_inf = [maximum(hk_inf[i, :]) - minimum(hk_inf[i, :]) for i ∈ collect(1:nn)]

    # pv_0_cdf = sort(vec(pv_0))
    pv_1_cdf = sort(vec(pv_1))
    pv_2_cdf = sort(vec(pv_2))
    pv_3_cdf = sort(vec(pv_3))
    pv_inf_cdf = sort(vec(pv_inf))
    y = collect(1:nn)./(nn)

    # define xlabel and x bounds
    xlabel = "Pressure range [m]" 
    xmin = 0
    xmax = 30
    # xmin = 0
    # xmax = 60
    # c = colors[1]
    # c = colors[2]
    c = colors[3]

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
        Coordinates(pv_inf_cdf, y)
    ), 
    LegendEntry(L"$\delta_{\infty}$")
    )
pgfsave("plots/"*net_name*"_range_cdf.pdf", cdf_plot)
pgfsave("plots/"*net_name*"_range_cdf.svg", cdf_plot)
pgfsave("plots/"*net_name*"_range_cdf.tex", cdf_plot; include_preamble=false)
cdf_plot
end



### objective time series plot ###
begin
    f_azp = hcat(f_azp, repeat([Inf], size(f_azp, 1)))
    f_scc = hcat(f_scc, repeat([Inf], size(f_scc, 1)))
end

begin
    # x = collect(0:0.25:24)
    x = collect(0:1:24)

    # c = colors[1]
    # c = colors[2]
    c = colors[3]

    # AZP objective plot
    azp_plot = @pgf Axis(
        {
            ylabel = "AZP [m]",
            # xlabel = {none},
            xmin = 0,
            xmax = 24,
            xtick = "{0, 4, ..., 24}",
            ymin = 15,
            ymax = 35,
            ytick = "{15, 20, ..., 35}",
            # ymin = 30,
            # ymax = 60,
            # ytick = "{30, 40, ..., 60}",
            # ymin = 32,
            # ymax = 48,
            # ytick = "{32, 36, ..., 48}",
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
        Coordinates(x, f_azp[2, :])
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
        Coordinates(x, f_azp[3, :])
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
        Coordinates(x, f_azp[4, :])
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
        Coordinates(x, f_azp[5, :])
    ), 
    LegendEntry(L"$\delta_{\infty}$"),
    VBand(
        {
            draw = {none},
            fill = cGrey,
            opacity = "0.2", 
            # mark_options = {"solid, fill_opacity=0.15"}
            }, 
            # (scc_time[1]-1)/4, (scc_time[end]-1)/4
            (scc_time[1]-1), (scc_time[end]-1)
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
            # ymax = 90,
            # ytick = "{10, 30, ..., 90}",
            ymin = 20,
            ymax = 100,
            ytick = "{20, 40, ..., 100}",
            # ymin = 5,
            # ymax = 45,
            # ytick = "{5, 15, ..., 45}",
            # tick_style = "black",
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
        Coordinates(x, f_scc[2, :])
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
        Coordinates(x, f_scc[3, :])
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
        Coordinates(x, f_scc[4, :])
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
        Coordinates(x, f_scc[5, :])
    ), 
    VBand(
        {
            draw = {none},
            fill = cGrey,
            opacity = "0.2", 
            # mark_options = {"solid, fill_opacity=0.15"}
            }, 
            # (scc_time[1]-1)/4, (scc_time[end]-1)/4
            (scc_time[1]-1), (scc_time[end]-1)
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

    pgfsave("plots/"*net_name*"_range_azp_scc.pdf", azp_scc_plot)
    pgfsave("plots/"*net_name*"_range_azp_scc.svg", azp_scc_plot)
    pgfsave("plots/"*net_name*"_range_azp_scc.tex", azp_scc_plot; include_preamble=false)
    azp_scc_plot
end



### residuals plot ###
begin

    # δ_idx = 1
    # δ_idx = 2
    δ_idx = 3


    c = blues[4:end]
    # c = greens[4:end]
    # c = reds[4:end]
    residuals[:, 1, :] .= nothing
    p_residual = residuals[:, :, δ_idx]; p_residual = map(x-> x === nothing ? Inf : x, p_residual)
    δ_name = L"\delta_{%$δ_idx}"
    
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
    num_iter = Int(ceil(num_iter/50)*50)

    # y-axis maximum
    ymax_p = Int(ceil(maximum(filter(isfinite, vec(p_residual)))*1.1))
    # ymax_p = 400
    if ymax_p ≥ 10
        scaled_y_ticks_p = "{base 10:-2}"
        y_tick_p = "{0, 10, ..., $ymax_p}"
    else
        scaled_y_ticks_p = "false"
        # y_tick_p = "{0, 20, ..., $ymax_p}"
    end
        


    # p_residual plot
    p_residual_plot = @pgf Axis(
        {
            ylabel = L"$\sqrt{n_n n_t} \times \|r\|_2$",
            xlabel = "Iteration",
            xmin = 0,
            # xmax = 40,
            xmax = num_iter,
            xtick = "{0, 25, ..., $num_iter}",
            # xtick = "{0, 10, ..., 40}",
            ymode = "log",
            ymax = 10^1,
            ymin = 0.005,
            # ytick = y_tick_p,
            tick_style = "black",
            # scaled_y_ticks = scaled_y_ticks_p,
            legend_pos = "outer north east",
            # scale_only_axis = true,
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            # y_tick_label_style = "{/pgf/number format/fixed zerofill}",
            # y_tick_label_style = "{/pgf/number format/fixed zerofill, /pgf/number format/precision=1, /pgf/number format/fixed}",
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
        HLine(
            { 
                style = "dotted, thick", 
                color = "black", 
                # color = colors[9],
                }, 
                10^-2
            ),
            # LegendEntry(L"$\epsilon$"),
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

    pgfsave("plots/"*net_name*"_range_"*string(δ[δ_idx])*"_residuals.pdf", p_residual_plot)
    pgfsave("plots/"*net_name*"_range_"*string(δ[δ_idx])*"_residuals.svg", p_residual_plot)
    pgfsave("plots/"*net_name*"_range_"*string(δ[δ_idx])*"_residuals.tex", p_residual_plot; include_preamble=false)
    p_residual_plot



end





### results plotting for standard ADMM experiments which do not converge ###

# load admm results
begin
    results_20 = load("data/admm_results/modena_range_delta_20_gamma_0.001_distributed.jld2")
    results_15 = load("data/admm_results/modena_range_delta_15_gamma_0.001_distributed.jld2")
    results_10 = load("data/admm_results/modena_range_delta_10_gamma_0.001_distributed.jld2")

    kmax = 1000
    residuals = Array{Union{Any}}(nothing, kmax, 3)
    residuals[:, 1] = results_20["p_residual"]
    residuals[:, 2] = results_15["p_residual"]
    residuals[:, 3] = results_10["p_residual"]

    residuals[1, :] .= nothing
    residuals = map(x-> x === nothing ? Inf : x, residuals)

end

# plotting code
begin
    c = greys

    residual_plot = @pgf Axis(
        {
            ylabel = L"$\sqrt{n_n n_t} \cdot \|r\|$",
            xlabel = "Iteration",
            xmin = 0,
            xmax = 1000,
            xtick = "{0, 250, ..., 1000}",
            ymode = "log",
            ymax = 10^1,
            ymin = 0.005,
            tick_style = "black",
            legend_pos = "north east",
            label_style = "{font=\\Large}",
            tick_label_style = "{font=\\large}",
            # y_tick_label_style = "{/pgf/number format/fixed zerofill}",
            # y_tick_label_style = "{/pgf/number format/fixed zerofill, /pgf/number format/precision=1, /pgf/number format/fixed}",
            # y_tick_label_style = "{/pgf/number format/precision=1}",
            legend_style = "{font=\\large}",
            title_style = "{font=\\Large}",
            title = L"$\rho=10^{-3}$"
        },
        PlotInc(
        {
            style = "solid, very thick",
            mark = "none",
            color = c[4],
        },
        Coordinates(collect(1:length(residuals[:, 1])), residuals[:, 1])
        ), 
        LegendEntry(L"$\delta_1$"),
        PlotInc(
            {
                style = "solid, very thick",
                mark = "none",
                color = c[6],
            },
            Coordinates(collect(1:length(residuals[:, 2])), residuals[:, 2])
            ), 
        LegendEntry(L"$\delta_2$"),
        PlotInc(
            {
                style = "solid, very thick",
                mark = "none",
                color = c[8],
            },
            Coordinates(collect(1:length(residuals[:, 3])), residuals[:, 3])
            ), 
            LegendEntry(L"$\delta_3$"),
        HLine(
            { 
                style = "dotted, thick", 
                color = "black", 
                # color = colors[9],
                }, 
                10^-2
            ),
            # LegendEntry(L"$\epsilon$"),
    )
    pgfsave("plots/modena_admm_residuals.pdf", residual_plot)
    pgfsave("plots/modena_admm_residuals.svg", residual_plot)
    pgfsave("plots/modena_admm_residuals.tex", residual_plot; include_preamble=false)
    residual_plot

end


