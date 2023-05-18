### old and obsolete code ###
#### objective value plot ###
# begin
#     @pgf obj_plot = Axis(
#         {
#             xmode = "log",
#             xlabel = L"\gamma",  # L"()" shows $()$ in latex math
#             ylabel = L"f",
#             # label_style = "{font=\\Large}",
#             # tick_label_style = "{font=\\large}",
#             # legend_style = "{font=\\large}",
#             ymin = 3220,
#             ymax = 3360,
#             ytick = "{3220, 3240, ..., 3360}",
#             tick_style = "black",
#             # ymin = 3220,
#             # ymax = 3290,
#             # ytick = "{3220, 3230, ..., 3290}",
#             # minor_y_tick_num = 10,
#             # legend_pos = "outer north east",
#         },
#         PlotInc(
#             {
#                 style = "solid, very thick",
#                 mark = "o",
#                 # mark_size = "3pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[5],
#                 # color = colors[1],
#                 # color = cTeal,
#                 color = colors[1],
#             },
#             Coordinates(γ_range, δ_p1[:, 2, 1])
#         ), 
#         # LegendEntry(L"$\delta_1 \, (\mathcal{P}_v)$"),
#         PlotInc(
#             {
#                 style = "dashed, very thick",
#                 mark = "o",
#                 # mark_size = "3pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[7],
#                 # color = colors[2],
#                 # color = cTeal,
#                 color = colors[1],
#             },
#             Coordinates(γ_range, δ_p1[:, 2, 2])
#         ), 
#         # LegendEntry(L"$\delta_2 \, (\mathcal{P}_v)$"),
#         PlotInc(
#             {
#                 style = "dotted, very thick",
#                 mark = "o",
#                 # mark_size = "3pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[9],
#                 # color = colors[3],
#                 # color = cTeal,
#                 color = colors[1],
#             },
#             Coordinates(γ_range, δ_p1[:, 2, 3])
#         ), 
#         # LegendEntry(L"$\delta_3 \, (\mathcal{P}_v)$"),
#         PlotInc(
#             {
#                 style = "solid, very thick",
#                 mark = "square",
#                 # mark_size = "2.5pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[5],
#                 # color = colors[1],
#                 # color = cRed,
#                 color = colors[2],
#             },
#             Coordinates(γ_range, δ_p2[:, 2, 1])
#         ), 
#         # LegendEntry(L"$\delta_1 \, (\mathcal{P}_r)$"),
#         PlotInc(
#             {
#                 style = "dashed, very thick",
#                 mark = "square",
#                 # mark_size = "2.5pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[7],
#                 # color = colors[2],
#                 # color = cRed,
#                 color = colors[2],
#             },
#             Coordinates(γ_range, δ_p2[:, 2, 2])
#         ), 
#         # LegendEntry(L"$\delta_2 \, (\mathcal{P}_r)$"),
#         PlotInc(
#             {
#                 style = "dotted, very thick",
#                 mark = "square",
#                 # mark_size = "2.5pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[9],
#                 # color = colors[3],
#                 # color = cRed,
#                 color = colors[2],
#             },
#             Coordinates(γ_range, δ_p2[:, 2, 3])
#         ), 
#         # LegendEntry(L"$\delta_3 \, (\mathcal{P}_r)$"),
#         # PlotInc(
#         #     {
#         #         style = "solid, very thick",
#         #         mark = "diamond",
#         #         mark_size = "2.5pt",
#         #         mark_options = {"solid, fill_opacity=0.15"},
#         #         # color = blues[5],
#         #         # color = colors[1],
#         #         # color = cRed,
#         #         color = cGrey,
#         #     },
#         #     Coordinates(γ_range, δ_p3[:, 2, 1])
#         # ), 
#         # # LegendEntry(L"$\delta_1$ (P3)"),
#         # PlotInc(
#         #     {
#         #         style = "dashed, very thick",
#         #         mark = "diamond",
#         #         mark_size = "2.5pt",
#         #         mark_options = {"solid, fill_opacity=0.15"},
#         #         # color = blues[7],
#         #         # color = colors[2],
#         #         # color = cRed,
#         #         color = cGrey,
#         #     },
#         #     Coordinates(γ_range, δ_p3[:, 2, 2])
#         # ), 
#         # # LegendEntry(L"$\delta_2$ (P3)"),
#         # PlotInc(
#         #     {
#         #         style = "dotted, very thick",
#         #         mark = "diamond",
#         #         mark_size = "2.5pt",
#         #         mark_options = {"solid, fill_opacity=0.15"},
#         #         # color = blues[9],
#         #         # color = colors[3],
#         #         # color = cRed,
#         #         color = cGrey,
#         #     },
#         #     Coordinates(γ_range, δ_p3[:, 2, 3])
#         # ), 
#         # # LegendEntry(L"$\delta_3$ (P3)"),
#         HLine(
#             { 
#                 style = "solid, very thick", 
#                 color = "black", 
#                 # color = colors[9],
#                 }, 
#                 sum(setdiff(f_azp_0, f_azp_0[scc_time])) + sum(f_scc_0[scc_time])*-1
#             ),
#             # LegendEntry(L"$\delta_{\infty}$"),
#     )
#     pgfsave("plots/"*net_name*"_obj.pdf", obj_plot)
#     pgfsave("plots/"*net_name*"_obj.svg", obj_plot)
#     pgfsave("plots/"*net_name*"_obj.tex", obj_plot; include_preamble=false)
#     obj_plot
# end

# ### no. of iterations plot ###
# begin
#     @pgf iter_plot = Axis(
#         {
#             xmode = "log",
#             # ymode = "log",
#             xlabel = L"\gamma",  # L"()" shows $()$ in latex math
#             ylabel = "Iterations",
#             ymin = 0,
#             ymax = 500,
#             tick_style = "black",
#             # ytick = "{0, 25, ..., 200}",
#             # legend_pos = "north east",
#             legend_pos = "outer north east",
#         },
#         PlotInc(
#             {
#                 style = "solid, very thick",
#                 mark = "o",
#                 # mark_size = "3pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[5],
#                 # color = colors[1],
#                 # color = cTeal,
#                 color = colors[1],
#             },
#             Coordinates(γ_range, δ_p1[:, 3, 1])
#         ), 
#         LegendEntry(L"$\delta_1 \, (\mathcal{P}_v)$"),
#         PlotInc(
#             {
#                 style = "dashed, very thick",
#                 mark = "o",
#                 # mark_size = "3pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[7],
#                 # color = colors[2],
#                 # color = cTeal,
#                 color = colors[1],
#             },
#             Coordinates(γ_range, δ_p1[:, 3, 2])
#         ), 
#         LegendEntry(L"$\delta_2 \, (\mathcal{P}_v)$"),
#         PlotInc(
#             {
#                 style = "dotted, very thick",
#                 mark = "o",
#                 # mark_size = "3pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[9],
#                 # color = colors[3],
#                 # color = cTeal,
#                 color = colors[1],
#             },
#             Coordinates(γ_range, δ_p1[:, 3, 3])
#         ), 
#         LegendEntry(L"$\delta_3 \, (\mathcal{P}_v)$"),
#         PlotInc(
#             {
#                 style = "solid, very thick",
#                 mark = "square",
#                 # mark_size = "2.5pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[5],
#                 # color = colors[1],
#                 # color = cRed,
#                 color = colors[2],
#             },
#             Coordinates(γ_range, δ_p2[:, 3, 1])
#         ), 
#         LegendEntry(L"$\delta_1 \, (\mathcal{P}_r)$"),
#         PlotInc(
#             {
#                 style = "dashed, very thick",
#                 mark = "square",
#                 # mark_size = "2.5pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[7],
#                 # color = colors[2],
#                 # color = cRed,
#                 color = colors[2],
#             },
#             Coordinates(γ_range, δ_p2[:, 3, 2])
#         ), 
#         LegendEntry(L"$\delta_2 \, (\mathcal{P}_r)$"),
#         PlotInc(
#             {
#                 style = "dotted, very thick",
#                 mark = "square",
#                 # mark_size = "2.5pt",
#                 mark_options = {"solid, fill_opacity=0.15"},
#                 # color = blues[9],
#                 # color = colors[3],
#                 # color = cRed,
#                 color = colors[2],
#             },
#             Coordinates(γ_range, δ_p2[:, 3, 3])
#         ), 
#         LegendEntry(L"$\delta_3 \, (\mathcal{P}_r)$"),
#         # PlotInc(
#         #     {
#         #         style = "solid, very thick",
#         #         mark = "diamond",
#         #         mark_size = "2.5pt",
#         #         mark_options = {"solid, fill_opacity=0.15"},
#         #         # color = blues[5],
#         #         # color = colors[1],
#         #         # color = cRed,
#         #         color = cGrey,
#         #     },
#         #     Coordinates(γ_range, δ_p3[:, 3, 1])
#         # ), 
#         # LegendEntry(L"$\delta_1$ (P3)"),
#         # PlotInc(
#         #     {
#         #         style = "dashed, very thick",
#         #         mark = "diamond",
#         #         mark_size = "2.5pt",
#         #         mark_options = {"solid, fill_opacity=0.15"},
#         #         # color = blues[7],
#         #         # color = colors[2],
#         #         # color = cRed,
#         #         color = cGrey,
#         #     },
#         #     Coordinates(γ_range, δ_p3[:, 3, 2])
#         # ), 
#         # LegendEntry(L"$\delta_2$ (P3)"),
#         # PlotInc(
#         #     {
#         #         style = "dotted, very thick",
#         #         mark = "diamond",
#         #         mark_size = "2.5pt",
#         #         mark_options = {"solid, fill_opacity=0.15"},
#         #         # color = blues[9],
#         #         # color = colors[3],
#         #         # color = cRed,
#         #         color = cGrey,
#         #     },
#         #     Coordinates(γ_range, δ_p3[:, 3, 3])
#         # ), 
#         # LegendEntry(L"$\delta_3$ (P3)"),
#         PlotInc(
#             {
#                 style = "solid, very thick",
#                 mark = "none",
#                 color = "black", 
#                 # color = colors[9],
#             },
#             Coordinates(γ_range, repeat([-1], length(γ_range)))
#         ), 
#             LegendEntry(L"$\delta_{\infty}$"),
#     )
#     pgfsave("plots/"*net_name*"_iter.pdf", iter_plot)
#     pgfsave("plots/"*net_name*"_iter.svg", iter_plot)
#     pgfsave("plots/"*net_name*"_iter.tex", iter_plot; include_preamble=false)
#     iter_plot
# end

# ### combine obj and iter plots ###
# begin
#     obj_iter_plot = @pgf GroupPlot(
#         { 
#             group_style = {
#                 group_size = "2 by 1",
#                 # xticklabels_at = "edge bottom",
#                 # yticklabels_at = "edge left",
#                 horizontal_sep = "1.5cm",
#                 },
#         },
#     obj_plot, iter_plot)
#     pgfsave("plots/"*net_name*"_obj_iter.pdf", obj_iter_plot)
#     pgfsave("plots/"*net_name*"_obj_iter.svg", obj_iter_plot)
#     pgfsave("plots/"*net_name*"_obj_iter.tex", obj_iter_plot; include_preamble=false)
#     obj_iter_plot
# end
