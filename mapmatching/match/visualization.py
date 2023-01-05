
import os
import time
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

try:
    from tilemap import plot_geodata, add_basemap
    TILEMAP_FLAG = True
except:
    def plot_geodata(data, *args, **kwargs):
        return data.plot(*args, **kwargs)
    TILEMAP_FLAG = False

def matching_debug_subplot(net, traj, item, level, src, dst, ax=None, maximun=None, legend=True, scale=.9, factor=4):
    """Plot the matching situation of one pair of od.

    Args:
        item (pandas.core.series.Series): One record in tList. The multi-index here is (src, dest).
        net ([type], optional): [description]. Defaults to net.
        ax ([type], optional): [description]. Defaults to None.
        legend (bool, optional): [description]. Defaults to True.

    Returns:
        ax: Ax.
    
    Example:
        matching_debug_subplot(graph_t.loc[1])
    """
    if ax is None:
        _, ax = plot_geodata(traj, scale=scale, alpha=.6, color='white')
    else:
        traj.plot(ax=ax, alpha=.6, color='white')
        ax.axis('off')
        if TILEMAP_FLAG:
            add_basemap(ax=ax, alpha=.5, reset_extent=False)
        # plot_geodata(traj, scale=scale, alpha=.6, color='white', ax=ax)

    # OD
    traj.loc[[level]].plot(ax=ax, marker="*", label=f'O ({src})', zorder=9)
    traj.loc[[item.pid_1]].plot(ax=ax, marker="s", label=f'D ({dst})', zorder=9)

    # path
    gpd.GeoDataFrame( item ).T.plot(ax=ax, color='red', label='path')
    net.get_edge([src]).plot(ax=ax, linestyle='--', alpha=.8, label=f'first({src})', color='green')
    net.get_edge([dst]).plot(ax=ax, linestyle=':', alpha=.8, label=f'last({dst})', color='black')

    # aux
    prob = item.observ_prob * item.trans_prob
    if 'dir_prob' in item:
        info = f"{prob:.3f} = {item.observ_prob:.2f} * {item.trans_prob:.2f} ({item.dist_prob:.2f}, {item.dir_prob:.2f})"
    else:
        info = f"{prob:.3f} = {item.observ_prob:.2f} * {item.trans_prob:.2f}"

    if maximun is not None and prob == maximun:
        color = 'red'
    elif maximun / prob < factor:
        color = 'blue'
    else:
        color = 'gray'
    ax.set_title(f"{src} -> {dst}: {info}", color = color)
    ax.set_axis_off()

    if legend: 
        ax.legend()
    
    return ax
    

def matching_debug_level(net, traj, df_layer, layer_id, debug_folder='./'):
    """PLot the matchings between levels (i, i+1)

    Args:
        traj ([type]): [description]
        tList ([type]): The candidate points.
        graph_t ([type]): [description]
        level ([type]): [description]
        net ([type]): [description]
        debug (bool, optional): [description]. Save or not

    Returns:
        [type]: [description]
    """

    rows = df_layer.index.get_level_values(0).unique()
    cols = df_layer.index.get_level_values(1).unique()
    n_rows, n_cols = len(rows), len(cols)

    _max = (df_layer.observ_prob * df_layer.trans_prob).max()
    
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    for i, src in enumerate(rows):
        for j, dst in enumerate(cols):
            ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1) 
            matching_debug_subplot(net, traj, df_layer.loc[src].loc[dst], layer_id, src, dst, ax=ax, maximun=_max)

    if 'dir_prob' in list(df_layer):
        _title = f'Level: {layer_id} [observ * trans (dis, dir)]'
    else:
        _title = f'Level: {layer_id} [observ * trans]'

    plt.suptitle(_title)
    plt.tight_layout()
    
    if debug_folder:
        t = time.strftime("%Y%m%d_%H", time.localtime()) 
        plt.savefig( os.path.join(debug_folder, f"{t}_level_{layer_id}.jpg"), dpi=300)
        plt.close()
        
    return True


def plot_matching(net, traj, cands, route, save_fn=None, satellite=True, column=None, categorical=True):
    def _base_plot(df):
        if column is not None and column in traj.columns:
            ax = df.plot(alpha=.3, column=column, categorical=categorical, legend=True)
        else:
            ax = df.plot(alpha=.3, color='black')
        ax.axis('off')
        
        return ax
    
    # plot，trajectory point
    _df = gpd.GeoDataFrame(pd.concat([traj, route]))
    if satellite:
        try:
            _, ax = plot_geodata(_df, alpha=0, tile_alpha=.5, reset_extent=True)
            if column is not None:
                traj.plot(alpha=0, column=column, categorical=categorical, legend=True, ax=ax)
        except:
            ax = _base_plot(_df)       
    else:
        ax = _base_plot(_df)
        
    traj.plot(ax=ax, color='blue', alpha=.5, label= 'Trajectory')
    traj.head(1).plot(ax=ax, marker = '*', color='red', zorder=9, label= 'Start point')
    
    # network
    edge_lst = net.spatial_query(box(*traj.total_bounds))
    net.get_edge(edge_lst).plot(ax=ax, color='black', linewidth=.8, alpha=.4, label='Network' )
    
    # candidate
    net.get_edge(cands.eid.values).plot(
        ax=ax, label='Candidates', color='blue', linestyle='--', linewidth=.8, alpha=.5)
    
    # route
    if route is not None:
        route.plot(ax=ax, label='Path', color='red', alpha=.5)
    
    ax.axis('off')
    if column is None:
        plt.legend(loc=1)
    
    if save_fn is not None:
        plt.tight_layout()
        plt.savefig(save_fn, dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close()
    
    return ax


def _base_plot(df, column=None, categorical=True):
    if column is not None and column in df.columns:
        ax = df.plot(alpha=.3, column=column, categorical=categorical, legend=True)
    else:
        ax = df.plot(alpha=.3, color='black')
    ax.axis('off')
    
    return ax


def plot_matching_result(traj_points, path, net, column=None, categorical=True):
    _df = gpd.GeoDataFrame(pd.concat([traj_points, path]))
    fig, ax = plot_geodata(_df, tile_alpha=.7, reset_extent=False, alpha=0)

    traj_points.plot(ax=ax, label='Trajectory', zorder=2, alpha=.5, color='b')
    traj_points.iloc[[0]].plot(ax=ax, label='Source', zorder=4, marker="*", color='orange')
    if path is not None:
        path.plot(ax=ax, color='r', label='Matched Path', zorder=3, linewidth=2, alpha=.7)

    x0, x1, y0, y1 = ax.axis()
    zones = gpd.GeoDataFrame({'geometry': [box(x0, y0, x1, y1)]})
    net.df_edges.sjoin(zones, how="inner", predicate='intersects')\
                .plot(ax=ax, color='black', label='roads', alpha=.3, zorder=2, linewidth=1)
    ax.legend(loc=1)

    return fig, ax
