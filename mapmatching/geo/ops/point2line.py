import numba
import numpy as np
import shapely
from shapely import Point, LineString
import geopandas as gpd
from geopandas import GeoDataFrame

from ..haversineDistance import haversine_geoseries, cal_coords_seq_distance


@numba.jit
def get_first_index(arr, val):
    """有效地返回数组中第一个值满足条件的索引
    Refs: https://blog.csdn.net/weixin_39707612/article/details/111457329;
    耗时： 0.279 us; np.argmax(arr> vak)[0] 1.91 us

    Args:
        A (np.array): Numpy arr
        k (float): value

    Returns:
        int: The first index that large that `val`
    """
    for i in range(len(arr)):
        if arr[i] >= val:
            return i

    return -1

def project_points_2_linestring(point:Point, line:LineString, normalized:bool=True, precision=1e-7):
    dist = line.project(point, normalized)
    proj_point = line.interpolate(dist, normalized)

    proj_point = shapely.set_precision(proj_point, precision)

    return proj_point, dist

def cut_linestring(line:LineString, offset:float, point:Point=None, normalized=True, cal_dist=True):
    _len = 1 if normalized else line.length
    coords = np.array(line.coords)

    if offset <= 0:
        res = {"seg_0": None, "seg_1": coords}
    elif offset >= _len:
        res = {"seg_0": coords, "seg_1": None}
    else:
        points = np.array([Point(*i) for i in coords])
        dist_intervals = line.project(points, normalized)

        idx = get_first_index(dist_intervals, offset)
        pd = dist_intervals[idx]
        if pd == offset:
            coords_0 = coords[:idx+1]
            coords_1 = coords[idx:]
        else:
            if point is None:
                point = line.interpolate(offset, normalized)
            cp = np.array(point.coords)
            coords_0 = np.concatenate([coords[:idx], cp]) 
            coords_1 = np.concatenate([cp, coords[idx:]]) 
        
        res = {'seg_0': coords_0, 'seg_1': coords_1}

    if cal_dist:
        res['len_0'] = cal_coords_seq_distance(res['seg_0'])[1] if res['seg_0'] is not None else 0
        res['len_1'] = cal_coords_seq_distance(res['seg_1'])[1] if res['seg_1'] is not None else 0

    return res

def test_cut_linestring(line, point):
    # test: project_point_2_linestring
    cp, dist = project_points_2_linestring(point, line)
    data = {'name': ['point', 'line', 'cp'],
            'geometry': [point, line, cp]
            }
    ax = gpd.GeoDataFrame(data).plot(column='name', alpha=.5)

    # test: cut_linestring
    seg_0, seg_1 = cut_linestring(line, dist)
    data = {'name': ['ori', 'seg_0', 'seg_1'],
            'geometry': [line, seg_0, seg_1]
            }
    gpd.GeoDataFrame(data).plot(column='name', legend=True, linestyle="--", ax=ax)

def project_points_2_linestrings(points:GeoDataFrame, lines:GeoDataFrame, 
                                 normalized:bool=True, drop_ori_geom=True, 
                                 keep_attrs:list=['eid', 'geometry'], precision=1e-7, 
                                 ll=True, cal_dist=True):
    """projects points to the nearest linestring

    Args:
        panos (GeoDataFrame | GeoSeries): Points
        paths (GeoDataFrame | GeoSeries): Edges
        keep_attrs (list, optional): _description_. Defaults to ['eid', 'geometry'].
        drop_ori_geom (bool, optional): Drop the origin point and line geometry. Defaults to True.

    Returns:
        GeoDataFrame: The GeoDataFrame of projected points with `proj_point`, `offset`
        
    Example:
        ```
        import geopandas as gpd
        from shapely import Point, LineString

        points = gpd.GeoDataFrame(
            geometry=[
                Point(113.93195659801206, 22.575930582940785),
                Point(113.93251505775076, 22.57563203614608),
                Point(113.93292030671412, 22.575490522559665),
                Point(113.93378178962489, 22.57534631453745)
            ]
        )

        lines = gpd.GeoDataFrame({
            "eid": [63048, 63935],
            "geometry": [
                LineString([(113.9319709, 22.5759509), (113.9320297, 22.5759095), (113.9321652, 22.5758192), (113.9323286, 22.575721), (113.9324839, 22.5756433), (113.9326791, 22.5755563), (113.9328524, 22.5754945), (113.9330122, 22.5754474), (113.933172, 22.5754073), (113.9333692, 22.5753782), (113.9334468, 22.5753503), (113.9335752, 22.5753413), (113.9336504, 22.5753383)]),
                LineString([(113.9336504, 22.5753383), (113.9336933, 22.5753314), (113.9337329, 22.5753215), (113.9337624, 22.5753098), (113.933763, 22.5753095)])]
        })

        prod_ps = project_points_2_linestrings(points.geometry, lines)
        _, ax = plot_geodata(prod_ps, color='red', label='proj', marker='*')
        lines.plot(ax=ax, label='lines')
        points.plot(ax=ax, label='points', alpha=.5)
        ax.legend()
        ```
    """
    proj_df = points.geometry.apply(lambda x: lines.loc[lines.distance(x).idxmin(), keep_attrs])\
                            .rename(columns={"geometry": 'edge_geom'})

    att_lst = ['proj_point', 'offset']
    proj_df.loc[:, 'point_geom'] = points.geometry
    proj_df.loc[:, att_lst] = proj_df.apply(
        lambda x: project_points_2_linestring(
            x.point_geom, x.edge_geom, normalized, precision), 
        axis=1, result_type='expand'
    ).values

    if not ll:
        cal_proj_dist = lambda x: x['query_geom'].distance(x['proj_point'])
        proj_df.loc[:, 'dist_p2c'] = proj_df.apply(cal_proj_dist, axis=1)
    else:
        proj_df.loc[:, 'dist_p2c'] = haversine_geoseries(
            proj_df['query_geom'], proj_df['proj_point'])

    if drop_ori_geom:
        proj_df.drop(columns=['point_geom', 'edge_geom'], inplace=True)

    return gpd.GeoDataFrame(proj_df).set_geometry('proj_point')


if __name__ == "__main__":
    line = LineString([(0, 0), (0, 1), (1, 1)])

    test_cut_linestring(line, Point((0.5, 0)))
    test_cut_linestring(line, Point((0, 1)))
    test_cut_linestring(line, Point((1.1, 1.5)))

