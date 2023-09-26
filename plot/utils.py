colors_dict = ['#C43129', '#43836A', '#E3A52C', '#1F919E']
pattern = [".", "x", "+", '/', '-', '\\']
markersymbol = [4, 7, 2]

# Common settings
# width = 700
# height = 240
linewidth = 2
markersize = 24
fontsize = 30
titlefontsize = 30


def bar_update_fig(fig, xtitle=None, ytitle=None, width=700, height=240, legendx=0.5, legendy=1.17):
    fig.update_layout(
        xaxis_title=xtitle, xaxis_title_font=dict(size=titlefontsize),
        yaxis_title=ytitle, yaxis_title_font=dict(size=titlefontsize),
        font=dict(size=fontsize, family='Arial', color='rgb(0, 0, 0)'),
        margin={'l': 15, 'r': 15, 't': 15, 'b': 15},
        width=width, height=height,
        legend=dict(
            yanchor="top", y=legendy,
            xanchor="center", x=legendx,
            orientation="h",
            font=dict(size=int(fontsize))
        ),
        legend_title="",
        plot_bgcolor='rgb(255,255,255)',
        xaxis=dict(
            showline=True,
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            ticks='inside',
            # tickformat='.2f',
            gridcolor='rgb(245, 245, 245)',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            # ticks='inside',
            tickformat='.1f',
            gridcolor='rgb(245, 245, 245)',
            mirror=True
        ),
    )
    return fig


def line_update_fig(fig, xtitle=None, ytitle=None, width=700, height=240, legendx=0.5, legendy=1.17):
    fig.update_layout(
        xaxis_title=xtitle, xaxis_title_font=dict(size=titlefontsize),
        yaxis_title=ytitle, yaxis_title_font=dict(size=titlefontsize),
        font=dict(size=fontsize, family='Arial', color='rgb(0, 0, 0)'),
        margin={'l': 15, 'r': 15, 't': 15, 'b': 15},
        width=width, height=height,
        legend=dict(
            yanchor="top", y=legendy,
            xanchor="center", x=legendx,
            orientation="h",
            font=dict(size=int(fontsize))
        ),
        legend_title="",
        plot_bgcolor='rgb(255,255,255)',
        xaxis=dict(
            showline=True,
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            ticks='inside',
            # tickformat='.2f',
            gridcolor='rgb(200, 200, 200)',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            ticks='inside',
            tickformat='.1f',
            gridcolor='rgb(200, 200, 200)',
            mirror=True
        ),
    )
    return fig
