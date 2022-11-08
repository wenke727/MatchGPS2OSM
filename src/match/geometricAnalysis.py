import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger
from shapely.geometry import box

from utils.timer import timeit
from geo.geo_helper import geom_series_distance, project_point_to_polyline

import warnings
warnings.filterwarnings('ignore')


def _plot_candidates(points, edges, match_res):
    ax = edges.plot()
    points.plot(ax=ax)

    res = gpd.GeoDataFrame(match_res)
    res.set_geometry('edge_geom', inplace=True)

    for _, group in res.groupby('pid'):
        group.plot(ax=ax, color='r', linewidth=3, alpha=.5, linestyle='--')
        
    return 


def _filter_candidate(df_candidates: gpd.GeoDataFrame,
                      top_k: int = 5,
                      pid: str = 'pid',
                      edge_keys: list = ['way_id', 'dir'],
                      level='info'
                      ):
    """Filter candidates, which belongs to the same way, and pickup the nearest one.

    Args:
        df_candidates (gpd.GeoDataFrame): _description_
        top_k (int, optional): _description_. Defaults to 5.
        pid (str, optional): _description_. Defaults to 'pid'.
        edge_keys (list, optional): The keys of edge ,which help to filter the edges which belong to the same road. Defaults to ['way_id', 'dir'].

    Returns:
        gpd.GeoDataFrame: The filtered candidates.
    """
    df = df_candidates
    origin_size = df.shape[0]

    df = df.sort_values([pid, 'dist_p2c'], ascending=[True, True])
    if edge_keys:
        df = df.groupby([pid] + edge_keys).head(1)

    df = df.groupby(pid).head(top_k).reset_index(drop=True)
    getattr(logger, level)(
        f"Top k candidate link, size: {origin_size} -> {df.shape[0]}")

    return df


def get_k_neigbor_edges(points: gpd.GeoDataFrame,
                        edges: gpd.GeoDataFrame,
                        top_k: int = 5,
                        radius: float = 50,
                        edge_keys: list = ['way_id', 'dir'],
                        edge_attrs: list = ['src', 'dst', 'way_id', 'dir', 'geometry'],
                        pid: str = 'pid',
                        eid: str = 'eid',
                        predicate: str = 'intersects',
                        ll: bool = True,
                        ll_to_utm_dis_factor=1e-5,
                        crs_wgs: int = 4326,
                        crs_prj: int = 900913):
    """Get candidates points and its localed edge for traj, which are line segment projection of p_i to these road segs.
    This step can be efficiently perfermed with the build-in grid-based spatial index.

    Args:
        points (gpd.GeoDataFrame): _description_
        edges (gpd.GeoDataFrame): _description_
        top_k (int, optional): _description_. Defaults to 5.
        radius (float, optional): _description_. Defaults to 50.
        edge_keys (list, optional): _description_. Defaults to ['way_id', 'dir'].
        edge_attrs (list, optional): _description_. Defaults to ['src', 'dst', 'way_id', 'dir', 'geometry'].
        pid (str, optional): _description_. Defaults to 'pid'.
        eid (str, optional): _description_. Defaults to 'eid'.
        predicate (str, optional): _description_. Defaults to 'intersects'.
        ll (bool, optional): _description_. Defaults to True.
        crs_wgs (int, optional): _description_. Defaults to 4326.
        crs_prj (int, optional): _description_. Defaults to 900913.

    Example:
        from shapely.geometry import Point, LineString
        
        # edges
        lines = [LineString([[0, i], [10, i]]) for i in range(0, 10)]
        lines += [LineString(([5.2,5.2], [5.8, 5.8]))]
        edges = gpd.GeoDataFrame({'geometry': lines, 
                                  'way_id':[i for i in range(10)] + [5]})
        # points
        a, b = Point(1, 1.1), Point(5, 5.1) 
        points = gpd.GeoDataFrame({'geometry': [a, b]}, index=[1, 3])
        
        # candidates
        res = get_candidates(points, edges, radius=2, top_k=2, ll=False, edge_keys=['way_id'])
        _plot_candidates(points, edges, res)
    
    Returns:
        _type_: _description_
    """
    if ll:
        radius *= ll_to_utm_dis_factor

    _edge_attrs = edge_attrs[:]
    edge_attrs = [i for i in edge_attrs if i in list(edges)]
    if len(edge_attrs) != len(_edge_attrs):
        logger.warning(f"Check edge attrs, only exists: {edge_attrs}")
    
    boxes = points.geometry.apply(lambda i: box(i.x - radius, i.y - radius, i.x + radius, i.y + radius))
    # The first subarray contains input geometry integer indexes.
    # The second subarray contains tree geometry integer indexes.
    cands = edges.sindex.query_bulk(boxes, predicate=predicate)
    if len(cands[0]) == 0:
        return None
    
    order_2_idx = {i: idx for i, idx in enumerate(points.index)}
    df_cand = pd.DataFrame.from_dict({pid: [order_2_idx[i] for i in cands[0]], 
                                      eid: edges.iloc[cands[1]].index})
    df_cand = df_cand.merge(points['geometry'], left_on=pid, right_index=True)\
                     .merge(edges[edge_attrs], left_on=eid, right_index=True)\
                     .rename(columns={'geometry_x': 'point_geom', 'geometry_y': 'edge_geom'})\
                     .sort_index()
    
    if ll:
        df_cand.loc[:, 'dist_p2c'] = geom_series_distance(df_cand.point_geom, df_cand.edge_geom, crs_wgs, crs_prj)
    else:
        df_cand.loc[:, 'dist_p2c'] = gpd.GeoSeries(df_cand.point_geom).distance(gpd.GeoSeries(df_cand.edge_geom))
        
    _df_cands = _filter_candidate(df_cand, top_k, pid, edge_keys)
    
    if _df_cands is None:
        logger.warning(f"Trajectory has no matching candidates")
        return None
    
    return _df_cands


def cal_observ_prob(dist, bias=0, deviation=20, normal=True):
    """The obervation prob is defined as the likelihood that a GPS sampling point `p_i` mathes a candidate point `C_ij`
    computed based on the distance between the two points. 

    Args:
        df (gpd.GeoDataFrame): Distance series or arrays.
        bias (float, optional): GPS measurement error bias. Defaults to 0.
        deviation (float, optional): GPS measurement error deviation. Defaults to 20.
        normal (bool, optional): Min-Max Scaling. Defaults to False.

    Returns:
        _type_: _description_
    """
    observ_prob_factor = 1 / (np.sqrt(2 * np.pi) * deviation)

    def f(x): return observ_prob_factor * \
        np.exp(-np.power(x - bias, 2)/(2 * np.power(deviation, 2)))

    _dist = f(dist)
    if normal:
        _dist /= _dist.max()

    return np.sqrt(_dist)


def project_point_to_line_segment(points, edges, keep_cols=['len_0', 'len_1', 'seg_0', 'seg_1']):
    def func(x): return project_point_to_polyline(
        x.points, x.edges, coord_sys=True
    )

    df = pd.DataFrame({'points': points, "edges": edges})
    res = df.apply(func, axis=1, result_type='expand')[keep_cols]

    return res

@timeit
def analyse_geometric_info(points: gpd.GeoDataFrame,
                           edges: gpd.GeoDataFrame,
                           top_k: int = 5,
                           radius: float = 50,
                           edge_keys: list = [],
                           edge_attrs: list = ['src', 'dst', 'way_id', 'dir', 'geometry'],
                           pid: str = 'pid',
                           eid: str = 'eid',
                           point_to_line_attrs: list = ['len_0', 'len_1', 'seg_0', 'seg_1'],
                           predicate: str = 'intersects',
                           ll: bool = True,
                           ll_to_utm_dis_factor=1e-5,
                           crs_wgs: int = 4326,
                           crs_prj: int = 900913,
                           ):
    cands = get_k_neigbor_edges(points, edges, top_k, radius, edge_keys,
                                edge_attrs, pid, eid, predicate, ll, 
                                ll_to_utm_dis_factor, crs_wgs, crs_prj)
    
    cands[point_to_line_attrs] = project_point_to_line_segment(
        cands.point_geom, cands.edge_geom, point_to_line_attrs)
    
    cands.loc[:, 'observ_prob'] = cal_observ_prob(cands.dist_p2c)
    
    return cands
    

if __name__ == "__main__":
    from shapely.geometry import Point, LineString
    
    # edges
    lines = [LineString([[0, i], [10, i]]) for i in range(0, 10)]
    lines += [LineString(([5.2,5.2], [5.8, 5.8]))]
    edges = gpd.GeoDataFrame({'geometry': lines, 
                              'way_id':[i for i in range(10)] + [5]})
    # points
    a, b = Point(1, 1.1), Point(5, 5.1) 
    points = gpd.GeoDataFrame({'geometry': [a, b]}, index=[1, 3])
    
    # candidates
    res = get_k_neigbor_edges(points, edges, radius=2, top_k=2, ll=False, edge_keys=['way_id'])
    _plot_candidates(points, edges, res)
    
