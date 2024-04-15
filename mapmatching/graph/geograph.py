import os
import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from shapely.geometry import LineString, box
# from networkx.classes import DiGraph

from .base import Digraph
from .astar import Astar
from .bi_astar import Bi_Astar
from ..osmnet.twoway_edge import parallel_offset_edge, swap_od
from ..utils.serialization import save_checkpoint, load_checkpoint


class GeoDigraph(Digraph):
    def __init__(self, df_edges:GeoDataFrame=None, df_nodes:GeoDataFrame=None, weight='dist',
                 crs_wgs:int=4326, crs_prj:int=None, ll:bool=False, *args, **kwargs):
        self.df_edges = df_edges
        self.df_nodes = df_nodes
        self.search_memo = {}
        self.nodes_dist_memo = {}
        self.ll = ll

        self.utm_crs = crs_prj
        self.wgs_crs = crs_wgs

        if df_edges is None or df_nodes is None:
            return
        
        if not ll:
            self.to_proj(refresh_distance=True)
        super().__init__(df_edges[['src', 'dst', weight]].sort_index().values, 
                             df_nodes.to_dict(orient='index'), *args, **kwargs)
        self.init_searcher()

    def init_searcher(self, algs='astar'):
        if algs == 'astar':
            self.searcher = Astar(self.graph, self.nodes, 
                                search_memo=self.search_memo, 
                                nodes_dist_memo=self.nodes_dist_memo,
                                max_steps=2000, max_dist=10000, ll=self.ll)
        else:
            self.searcher = Bi_Astar(self.graph, self.graph_r, self.nodes,
                                    search_memo=self.search_memo,
                                    nodes_dist_memo=self.nodes_dist_memo
                                    )

    def search(self, src, dst, max_steps=2000, max_dist=10000, coords=True):
        # keys: status, vpath, epath, cost, geometry
        route = self.searcher.search(src, dst, max_steps, max_dist)
        
        if 'epath' not in route:
            epath = self.transform_vpath_to_epath(route['vpath'])
            route['epath'] = epath
            if epath is not None:
                _df = self.get_edge(epath)
                _sum = _df.dist.fillna(0).sum()
                route['dist'] = _sum
                if _sum == 0:
                    route['avg_speed'] = np.average(_df.speed.values)
                else:
                    route['avg_speed'] = np.average(_df.speed.values, weights=_df.dist.values)
            else:
                route['avg_speed'] = 0
                route['dist'] = 0
                
        if coords and 'coords' not in route:
            lst = route['epath']
            if lst is None:
                route['coords'] = None
            else:
                route['coords'] = self.transform_epath_to_coords(lst)
        
        return route

    def spatial_query(self, geofence, name='df_edges', predicate='intersects'):
        gdf = getattr(self, name)
        
        idxs = gdf.sindex.query(geofence, predicate=predicate)
        
        return gdf.iloc[idxs].index

    """ get attribute """
    def get_edge(self, eid, attrs=None, reset_index=False):
        """Get edge by eid [0, n]"""
        res = self._get_feature('df_edges', eid, attrs, reset_index)
        # logger.debug(f"\n{res}")

        return res

    def get_node(self, nid, attrs=None, reset_index=False):
        return self._get_feature('df_nodes', nid, attrs, reset_index)

    def _get_feature(self, df_name, id, attrs=None, reset_index=False):
        """get edge by id.

        Args:
            id (_type_): _description_
            att (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        df = getattr(self, df_name)
        if df is None:
            print(f"Don't have the attibute {df_name}")
            return None

        if isinstance(id, int) and id not in df.index:
            return None

        if isinstance(id, list) or isinstance(id, tuple) or isinstance(id, np.ndarray):
            for i in id:
                if i not in df.index:
                    return None
        
        res = df.loc[id]
        
        if attrs is not None:
            res = res[attrs]
        
        if reset_index:
            res.reset_index(drop=True, inplace=True)
        
        return res

    def get_pred_edges(self, eid):
        src, _ = self.eid_2_od[eid]
        eids = [i['eid'] for i in self.graph_r[src].values()]

        return self.df_edges.loc[eids]

    def get_succ_edges(self, eid):
        _, dst = self.eid_2_od[eid]
        eids = [i['eid'] for i in self.graph[dst].values()]

        return self.df_edges.loc[eids]

    def get_way(self, way_id):
        df = self.df_edges.query("way_id == @way_id")
        if df.shape[0] == 0:
            return None
        
        return df

    """ transfrom """
    def transform_epath_to_coords(self, eids):
        steps = self.get_edge(eids, attrs=['geometry'], reset_index=True)
        coords = np.concatenate(steps.geometry.apply(lambda x: x.coords), axis=0)
        
        return coords           
           
    def transform_epath_to_linestring(self, eids):
        return LineString(self.transform_epath_to_coords(eids))

    """ io """
    def to_postgis(self, name, nodes_attrs=['nid', 'x', 'y', 'traffic_signals', 'geometry']):
        from ..geo.io import to_postgis
        df_node_with_degree = self.df_nodes.merge(self.calculate_degree(), left_index=True, right_index=True).reset_index()
        
        to_postgis(self.df_edges, f'topo_osm_{name}_edge', if_exists='replace')
        to_postgis(df_node_with_degree, f'topo_osm_{name}_endpoint', if_exists='replace')
        
        self.df_nodes.loc[:, 'nid'] = self.df_nodes.index
        nodes_attrs = [i for i in nodes_attrs if i in list(self.df_nodes) ]
        self.df_nodes = self.df_nodes[nodes_attrs]
        to_postgis(self.df_nodes, f'topo_osm_{name}_node', if_exists='replace')

        return True

    def to_csv(self, name, folder = None):
        edge_fn = f'topo_osm_{name}_edge.csv'
        node_fn = f'topo_osm_{name}_endpoint.csv'
        
        if folder is not None:
            edge_fn = os.path.join(edge_fn, edge_fn)
            node_fn = os.path.join(edge_fn, node_fn)
        
        df_edges = self.df_edges.copy()
        atts = ['eid', 'rid', 'name', 's', 'e', 'order', 'road_type', 'dir', 'lanes', 'dist', 'oneway', 'geom_origin']
        pd.DataFrame(df_edges[atts].rename({'geom_origin': 'geom'})).to_csv(edge_fn, index=False)
        
        df_nodes = self.df_nodes.copy()
        df_nodes.loc[:, 'nid'] = df_nodes.index
        df_nodes.loc[:, 'geom'] = df_nodes.geometry.apply(lambda x: x.to_wkt())
        
        atts = ['nid', 'x', 'y', 'traffic_signals', 'geometry']
        pd.DataFrame(df_nodes[atts].rename({'geom_origin': 'geom'})).to_csv(node_fn, index=False)
        
        return True

    def save_checkpoint(self, ckpt):
        return save_checkpoint(self, ckpt)

    def load_checkpoint(self, ckpt):
        from loguru import logger
        
        load_checkpoint(ckpt, self)
        self.logger = logger

        return self

    """ adding and removing nodes and edges """
    def add_edge(self, start, end, length=None):
        # add edge to dataframe
        return super().add_edge(start, end, length)
    
    def add_reverse_way(self, way_id, od_attrs=['src', 'dst'], offset=True):
        df_edges_rev = self.df_edges.query('way_id == @way_id')
        if df_edges_rev.shape[0] == 0:
            print(f"check way id {way_id} exist or not.")
            return False
        if df_edges_rev.dir.nunique() >= 2:
            return False

        ring_mask = df_edges_rev.geometry.apply(lambda x: x.is_ring)
        df_edges_rev = df_edges_rev[~ring_mask]
        df_edges_rev = swap_od(df_edges_rev)

        self.add_edges_from_df(df_edges_rev)

        # two ways offsets
        if offset:
            idxs = self.df_edges.query('way_id == @way_id').index
            self.df_edges.loc[idxs, 'geom_origin'] = self.df_edges.loc[idxs].geometry.copy()
            self.df_edges.loc[idxs, 'geometry'] = self.df_edges.loc[idxs].apply( lambda x: parallel_offset_edge(x), axis=1 )

        self.search_memo.clear()

        return True

    def add_edges_from(self, edges):
        return super().build_graph(edges)

    def add_edges_from_df(self, df):
        eids = range(self.max_eid, self.max_eid + df.shape[0])
        df.index = df.loc[:, 'eid'] = eids

        df_edges = gpd.GeoDataFrame(pd.concat([self.df_edges, df]))
        self.df_edges = df_edges

        return self.add_edges_from(df[['src', 'dst', 'dist']].values) 

    def remove_edge(self, eid=None, src=None, dst=None):
        assert eid is not None or src is not None and dst is not None

        if eid is None:
            eid = self.get_eid(src, dst)
        else:
            src, dst = self.eid_2_od[eid]

        self.df_edges.drop(index=eid, inplace=True)
        self.search_memo.clear()

        return super().remove_edge(src, dst)

    """ coord system """
    def align_crs(self, gdf):
        """
        Aligns the coordinate reference system (CRS) of a given GeoDataFrame to the CRS of the current object.

        This method checks if the CRS of the provided GeoDataFrame (`gdf`) matches the CRS of the current object (`self.crs`). 
        If they match, the original GeoDataFrame is returned without any modification. If they do not match, 
        the GeoDataFrame is transformed to the same CRS as the current object and then returned.

        Parameters:
        - gdf (gpd.GeoDataFrame): The GeoDataFrame whose CRS is to be aligned.

        Returns:
        - gpd.GeoDataFrame: The GeoDataFrame with CRS aligned to `self.crs`.

        Example:
        >>> current_crs_gdf = MyClass(crs="EPSG:4326")
        >>> new_gdf = gpd.read_file("path/to/geodatafile.shp")
        >>> aligned_gdf = current_crs_gdf.align_crs(new_gdf)
        """
        
        assert gdf.crs is not None
        assert self.utm_crs is not None
        
        gdf.to_crs(self.epsg, inplace=True)

        return gdf

    def convert_to_wgs(self, gdf):
        assert gdf.crs is not None
        gdf.to_crs(self.wgs_crs, inplace=True)

        return gdf

    def to_ll(self):
        self.df_edges.to_crs(self.wgs_crs, inplace=True)
        self.df_nodes.to_crs(self.wgs_crs, inplace=True)
        
        return True

    def to_proj(self, refresh_distance=False, eps=1e-5):
        if self.utm_crs is None:
            self.utm_crs = self.df_edges.estimate_utm_crs().to_epsg()

        self.df_edges.to_crs(self.utm_crs, inplace=True)
        if refresh_distance:
            mask = self.df_edges.geometry.apply(lambda x: not x.is_empty)
            self.df_edges.loc[mask, 'dist'] = self.df_edges[mask].length
            
            mask = self.df_edges.loc[:, 'dist'] == 0
            self.df_edges.loc[mask, 'dist'] = eps
            
        self.df_nodes.to_crs(self.utm_crs, inplace=True)

        return True

    @property
    def crs(self):
        return self.df_edges.crs
    
    @property
    def epsg(self):
        return self.df_edges.crs.to_epsg()

    def add_edge_map(self, ax, crs=4326, show_node=True, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        x0, x1, y0, y1 = ax.axis()
        zones = gpd.GeoDataFrame({'geometry': [box(x0, y0, x1, y1)]})

        if crs == 4326:
            if not hasattr(self, "df_edges_ll"):
                self.df_edges_ll = self.df_edges.to_crs(self.wgs_crs)
            df_edges = self.df_edges_ll
            if show_node:
                if not hasattr(self, 'df_nodes_ll'):
                    self.df_nodes_ll = self.df_nodes.to_crs(self.wgs_crs)
                df_nodes = self.df_nodes_ll
        else:
            df_edges = self.df_edges
            if show_node :
                df_nodes = self.df_nodes
        df_edges = df_edges.sjoin(zones, how="inner", predicate='intersects')
        if df_edges.empty:
            return ax
        
        df_edges.plot(ax=ax, *args, **kwargs)
        if show_node:
            df_nodes.plot(ax=ax, *args, **kwargs, facecolor='white', markersize=6)

        return ax

if __name__ == "__main__":
    network = GeoDigraph()
    network.load_checkpoint(ckpt='./cache/Shenzhen_graph_9.ckpt')

    route = network.search(src=7959990710, dst=499265789)
    # route = network.search(src=7959602916, dst=7959590857)

    network.df_edges.loc[route['epath']].plot()
    
    network.transform_epath_to_linestring(route['epath'])
