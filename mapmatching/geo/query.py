import shapely
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import geometry as shapely_geom

from .ops.linear_referencing import compute_point_to_line_proximity
from ..utils import timeit

def check_sindex(gdf: GeoDataFrame):
    # Ensure spatial index is built
    if not gdf.has_sindex:
        try:
            gdf.sindex
        except:
            raise ValueError()

def extract_candidate_geometries(query_gdf, target_gdf, candidates, query_id_col):
    """
    Extract and annotate geometries from a target GeoDataFrame based on candidate indices derived
    from a spatial index query.

    Parameters:
    - query_gdf (GeoDataFrame): The GeoDataFrame containing query geometries.
    - target_gdf (GeoDataFrame): The GeoDataFrame from which to retrieve candidate geometries.
    - candidates (tuple): A tuple of arrays (query_indices, target_indices) indicating the indices
      of matching geometries in the query and target GeoDataFrames.
    - query_id_col (str): The column name to assign to the query indices in the returned DataFrame.

    Returns:
    - GeoDataFrame: A new GeoDataFrame with the candidate geometries, including metadata about the
      original query geometries.
    """
    # Retrieve the geometries using the indices provided by candidates
    query_indices, target_indices = candidates
    query_points = query_gdf.iloc[query_indices]
    cand_geometries = target_gdf.iloc[target_indices]
    
    # Rename columns and add new columns to annotate the results
    cand_geometries.rename(columns={'geometry': 'edge_geom'}, inplace=True)
    cand_geometries.loc[:, query_id_col] = query_points.index
    cand_geometries.loc[:, "query_geom"] = query_points.values

    return cand_geometries

def project_query_on_candidates(df_cands, project=True):
    # dist_p2c
    if not project:
        cal_proj_dist = lambda x: x['query_geom'].distance(x['edge_geom'])
        df_cands.loc[:, 'dist_p2c'] = df_cands.apply(cal_proj_dist, axis=1)

        return df_cands

    df_projs = compute_point_to_line_proximity(df_cands['query_geom'], df_cands['edge_geom'])
    df_cands.loc[:, df_projs.keys()] = df_projs.values()
    # df_cands = gpd.GeoDataFrame(df_cands, crs=gdf.crs, geometry='proj_point')

    return df_cands
    
def visualize_query_and_candidates(cands):
    # TODO draw buffer
    from ..geo.vis import plot_geodata
    _, ax = plot_geodata(cands, color='r', tile_alpha=.6, alpha=0)

    cands.set_geometry('edge_geom').plot(ax=ax, column='dist_p2c', cmap='Reds_r', legend='candidates')
    if 'proj_point' in list(cands):
        cands.loc[:, 'proj_point'] = cands['proj_point'].apply(shapely.Point)
        cands.set_geometry('proj_point').plot(ax=ax, cmap='Reds_r')
    cands.set_geometry('query_geom').plot(ax=ax, marker='*', label='Point', zorder=9)

    return ax

def filter_top_k_candidates(df: gpd.GeoDataFrame,
                      pid: str = 'pid',
                      top_k: int = 5,
                      ):
    """Filter candidates, which belongs to the same way, and pickup the nearest one.

    Args:
        df (gpd.GeoDataFrame): df candidates.
        top_k (int, optional): _description_. Defaults to 5.
        pid (str, optional): _description_. Defaults to 'pid'.

    Returns:
        gpd.GeoDataFrame: The filtered candidates.
    """
    # origin_size = df.shape[0]
    df = df.sort_values([pid, 'dist_p2c'])\
           .groupby(pid)\
           .head(top_k)\
           .reset_index(drop=True)

    return df

def prepare_query_object(query, query_id, gdf_crs):
    """
    Prepare the query object for spatial querying.

    Args:
        query: The query object (GeoDataFrame, GeoSeries, or geometry).
        query_id: The identifier for the query object.
        gdf_crs: The coordinate reference system of the base GeoDataFrame.

    Returns:
        GeoSeries: The prepared query object.
    """
    if isinstance(query, shapely_geom.base.BaseGeometry):
        _query = gpd.GeoSeries([query])
    elif isinstance(query, GeoDataFrame):
        _query = query.set_index(query_id)['geometry'] if query_id in list(query) else query['geometry']
        _query.index.set_names(query_id, inplace=True)
    elif isinstance(query, gpd.GeoSeries):
        _query = query.copy()
        _query.index.set_names(query_id, inplace=True)
    else:
        raise TypeError("Query object type is not supported.")

    if _query.crs != gdf_crs:
        _query = _query.to_crs(gdf_crs)

    return _query

def query_spatial_index(query_objects, gdf, radius, predicate):
    """
    Perform spatial indexing query.

    Args:
        gdf: Base GeoDataFrame with spatial index.
        query_objects: Prepared query objects.
        radius: Search radius.
        predicate: Spatial predicate for querying (e.g., "intersects").

    Returns:
        Tuple: Indices of matched geometries in gdf.
    """
    get_box = lambda geom: shapely_geom.box(geom.x - radius, geom.y - radius, geom.x + radius, geom.y + radius)
    query_boxes = query_objects.apply(get_box)
    
    return gdf.sindex.query_bulk(query_boxes, predicate)

def process_query_results(query_objects, gdf, cands, query_id, project):
    """
    Process the results of a spatial query.

    Args:
        query_objects: The query objects.
        gdf: Base GeoDataFrame.
        cands: Indices of matched geometries.
        query_id: Identifier for the query object.
        project: Whether to project query objects onto gdf geometries.

    Returns:
        GeoDataFrame: DataFrame of candidates with additional info.
    """
    df_cands = extract_candidate_geometries(query_objects, gdf, cands, query_id)
    project_query_on_candidates(df_cands, project)
    
    return df_cands

def find_nearest_geometries(query_point: GeoDataFrame, gallery_geoms: GeoDataFrame, query_id='qid', 
                            max_distance: float = 50, top_k=None, predicate: str = 'intersects', 
                            check_diff=True, project=True, keep_geom=True):
    # TODO: Determine appropriate index for gdf
    check_sindex(gallery_geoms)
    _query = prepare_query_object(query_point, query_id, gallery_geoms.crs)
    
    # Perform spatial indexing query
    cands = query_spatial_index(_query, gallery_geoms, max_distance, predicate)
    if len(cands[0]) == 0:
        return None, None

    # Process query results
    # df_cands = process_query_results(_query, gallery_geoms, cands, query_id, project)
    df_cands = extract_candidate_geometries(_query, gallery_geoms, cands, query_id)
    df_cands = project_query_on_candidates(df_cands, project)

    # Further filtering and sorting
    if max_distance:
        df_cands.query(f"dist_p2c <= {max_distance}", inplace=True)
    if top_k:
        df_cands = filter_top_k_candidates(df_cands, query_id, top_k)

    if not keep_geom:
        df_cands.drop(columns=["query_geom", "edge_geom"], inplace=True)

    # Check difference
    no_cands_query = None
    if check_diff:
        cands_pid = set(cands[0])
        all_pid = set(_query.index.unique())
        no_cands_query = all_pid.difference(cands_pid)
        warnings.warn(f"{no_cands_query} has no neighbors within the {max_distance} search zone.")

    return df_cands.set_geometry('edge_geom').set_crs(gallery_geoms.crs), no_cands_query

if __name__ == "__main__":
    pass

    
