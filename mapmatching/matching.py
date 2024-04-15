import os
import numpy as np
import geopandas as gpd
from shapely import box
from copy import deepcopy
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt

from .graph import GeoDigraph
from .update_network import check_steps
from .geo.metric import lcss, edr, erp
from .geo.ops import check_duplicate_points
from .geo.ops.simplify import simplify_trajetory_points
from .geo.ops.resample import resample_polyline_seq_to_point_seq, resample_point_seq


from .match.status import STATUS
from .match.io import load_points
from .match.geometricAnalysis import query_candidates
from .match.candidatesGraph import CandidateGraphConstructor
from .match.viterbi import find_matched_sequence
from .match.postprocess import get_path, project, transform_mathching_res_2_path
from .match.visualization import plot_matching_result, debug_traj_matching

from .utils.timer import timeit
from .utils.logger_helper import make_logger
from .setting import DATA_FOLDER, DEBUG_FOLDER


class ST_Matching():
    def __init__(self, net: GeoDigraph,
                 max_search_steps = 2000, max_search_dist = 10000, prob_thres = .8, ll = False,
                 log_folder = './log', console = True,
        ):
        self.net = net
        edge_attrs = ['eid', 'src', 'dst', 'way_id', 'dir', 'dist', 'speed', 'geometry']
        # Avoid waste time on created new objects by slicing
        self.base_edges = self.net.df_edges[edge_attrs]
        self.base_edges.sindex

        self.crs_wgs = 4326
        self.utm_crs = self.net.utm_crs
        self.ll = ll
        
        self.debug_folder = DEBUG_FOLDER
        self.logger = make_logger(log_folder, console=console, level="INFO")
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

        # hyper parameters
        self.prob_thres = prob_thres
        self.route_planning_max_search_steps = max_search_steps
        self.route_planning_max_search_dist = max_search_dist
        
        self.set_search_candidates_variables()
    
    def set_search_candidates_variables(self, top_k = 5, search_radius = 50, loc_bias = 0, loc_deviaction = 100):
        self.top_k = top_k
        self.search_radius = search_radius
        self.loc_bias = loc_bias
        self.loc_deviaction = loc_deviaction
        
    def preprocess_traj(self, traj, simplify, tolerance, check_duplicate):    
        _traj = self.align_crs(traj.copy())
        if simplify:
            _traj = self.simplify(_traj, tolerance = tolerance)
        elif check_duplicate:
            _traj = check_duplicate_points(_traj)
            
        return _traj

    def construct_graph(self, cands, points, dir_trans):
        return CandidateGraphConstructor.construct_graph(cands, points, dir_trans)

    @timeit
    def matching(self, traj, dir_trans=False, 
                 simplify=True, tolerance=5, check_duplicate=False, 
                 plot=False, save_fn=None,
                 debug_in_levels=False, details=False, metric=None, 
                 check_topo=False, beam_search=True,
                 verbose=False):
        self.logger.trace("start")
        res = {'status': STATUS.UNKNOWN, 'ori_crs': deepcopy(traj.crs.to_epsg()), "probs": {}}
        
        # 1. preprocess
        pts = self.preprocess_traj(traj, simplify, tolerance, check_duplicate)  
        
        # 2. geometric analysis
        cands = query_candidates(pts, self.base_edges, self.top_k, self.search_radius, 
            bias = self.loc_bias, deviation = self.loc_deviaction
        )
        s, _ = self._is_valid_cands(pts, cands, res)
        if not s: 
            return res

        # 3. spatial analysis
        # rList, graph = self.spatial_analysis(pts, cands, dir_trans, metric=res['probs'])
        graph = self.construct_graph(cands, traj, dir_trans=dir_trans)
        matching_score, rList, graph = find_matched_sequence(cands, graph, self.net, dir_trans)
        res['probs']['prob'] = matching_score
        
        match_res, steps = self.get_mathcing_path(rList, graph, cands, metric=res['probs'], prob_thres=self.prob_thres)
        
        if 'status' in res['probs']:
            res['status'] = res['probs']['status']
            del res['probs']['status']
        res.update(match_res)

        # 4. metric
        if metric is not None:
            res[metric] = self.eval(pts, res, metric=metric)
            self.logger.debug(f"{metric}: {res['metric']}")

        # 5. add details
        if details or check_topo:
            res['details'] = {
                "simplified_traj": pts,
                'cands': cands, 
                'rList': rList, 
                "steps": steps, 
                'graph': self.get_matching_graph(graph), 
                'path': self.transform_res_2_path(res), 
            }

        # plot
        if plot or save_fn:
            fig, ax = self.plot_result(traj, res)
            if plot:
                plt.show()
            else:
                plt.close()
            if save_fn:
                fig.savefig(save_fn, dpi=300, bbox_inches='tight', pad_inches=0.02)

        # debug helper
        if debug_in_levels:
            self.debug_matching_among_levels(pts, graph)

        # check topo        
        if check_topo:
            flag = check_steps(self, res, prob_thred=.75, factor=1.2)
            if flag:
                res = self.matching(pts, dir_trans, beam_search,
                 False, tolerance, plot, save_fn, debug_in_levels, details, metric, 
                 check_duplicate, False)

        return res

    def _is_valid_cands(self, traj, cands, info, eps = 1e-7):
        # -> status, route
        if cands is None or isinstance(cands, gpd.GeoDataFrame) and cands.empty:
            info['status'] = STATUS.NO_CANDIDATES
            info['probs'] = {'norm_prob': 0}
            edges_box = box(*self.base_edges.total_bounds)
            traj_box = box(*traj.total_bounds)
            flag = edges_box.contains(traj_box)
            if not flag:
                details = "Please adjust the bbox to contain the trajectory."
                info['detail'] = details
                self.logger.error(details)
            assert flag, "check the bbox contained the trajectory or not"

            return False, None
        
        # Only one single point matched
        if traj.shape[0] == 1 or cands.pid.nunique() == 1: 
            eid = cands.sort_values('dist_p2c').head(1).eid.values
            coord = cands.iloc[0]['proj_point']
            res = {'epath': eid, 'step_0': [coord, [coord[0] + eps, coord[1] + eps]]}
            info.update(res)
            info['status'] = STATUS.ONE_POINT
            info['probs'] = {'norm_prob': 0}
            
            return False, res
        
        return True, None

    def _spatial_analysis(self, traj, cands, dir_trans, metric={}):
        graph = construct_graph(traj, cands, dir_trans=dir_trans)
        map_matching_score, rList, graph = find_matched_sequence(cands, graph, self.net, dir_trans)
        # graph = analyse_spatial_info(self.net, traj, cands, dir_trans)
        # map_matching_score, rList = process_viterbi_pipeline(cands, graph[['pid_1', 'dist_prob']])

        metric['prob'] = map_matching_score

        return rList, graph

    def eval(self, traj, res=None, path=None, resample=5, eps=10, metric='lcss', g=None):
        """
        lcss 的 dp 数组 循环部分，使用numba 加速，这个环节可以降低 10% 的时间消耗（20 ms） 
        """
        assert res is not None or path is not None
        assert metric in ['lcss', 'edr', 'erp']
        
        if path is None:
            path = self.transform_res_2_path(res)

        if traj.crs.to_epsg() != path.crs.to_epsg():
            traj = traj.to_crs(path.crs.to_epsg())

        if resample:
            _, path_coords_np = resample_polyline_seq_to_point_seq(path.geometry, step=resample,)
            _, traj_coords_np = resample_point_seq(traj.geometry, step=resample)
        else:
            path_coords_np = np.concatenate(path.geometry.apply(lambda x: x.coords[:]).values)
            traj_coords_np = np.concatenate(traj.geometry.apply(lambda x: x.coords[:]).values)
            
        eval_funs = {
            'lcss': [lcss, (traj_coords_np, path_coords_np, eps, self.ll)], 
            'edr': [edr, (traj_coords_np, path_coords_np, eps)], 
            'edp': [erp, (traj_coords_np, path_coords_np, g)]
        }
        _eval = eval_funs[metric]

        return _eval[0](*_eval[1])

    def project(self, points, path, keep_attrs=['eid', 'proj_point'], normalized=True, reset_geom=True):
        return project(points, path, keep_attrs, normalized, reset_geom)

    def load_points(self, fn, simplify=False, tolerance: int = 10,
                    crs: int = None, in_sys: str = 'wgs', out_sys: str = 'wgs'):
        traj, _ = load_points(fn, simplify, tolerance, crs, in_sys, out_sys)
        
        return traj

    def simplify(self, points:gpd.GeoDataFrame, tolerance:int=None, inplace=False):        
        return simplify_trajetory_points(points, tolerance, inplace=True, logger=self.logger)

    def update(self):
        return NotImplementedError
    
    def align_crs(self, traj):
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
        
        return self.net.align_crs(traj)

    """ get helper """
    def transform_res_2_path(self, res, ori_crs=True, attrs=None):
        return transform_mathching_res_2_path(res, self.net, ori_crs, attrs)

    def get_edges(self, eids, attrs=None, reset_index=False):
        """get edge by eid list"""
        return self.net.get_edge(eids, attrs, reset_index)

    def get_nodes(self, nids, attrs=None, reset_index=False):
        """get node by nid list"""
        return self.net.get_node(nids, attrs, reset_index)

    def get_mathcing_path(self, rList: GeoDataFrame, graph: GeoDataFrame, cands: GeoDataFrame, 
                          metric: dict={}, prob_thres: float=.8):
        return get_path(rList, graph, cands, metric, prob_thres)

    def get_utm_crs(self):
        return self.utm_crs

    def get_matching_graph(self, graph, 
                           attrs = ['pid_1', 'step_0_len', 'step_n_len', 'cost', 'sp_dist', 'euc_dist', 'dist_prob', 
                                    'trans_prob', 'observ_prob', 'prob', 'flag', 'status', 'dst', 'src','step_0', 
                                    'geometry', 'step_n', 'path', 'epath', 'vpath', 'dist', 'dist_0', 'step_1']):
        attrs = [i for i in attrs if i in list(graph)]
        if 'move_dir' in graph:
            attrs += ['move_dir']
        
        return graph[attrs]

    """ visualization """
    def debug_matching_among_levels(self, traj: gpd.GeoDataFrame, graph: gpd.GeoDataFrame, 
                                    level: list=None, debug_folder: str='./debug'):

        return debug_traj_matching(traj, graph, self.net, level, debug_folder)

    def plot_result(self, traj, info, filter_key='prob', legend=True):
        info = deepcopy(info)
        if info['status'] == 3:
            path = None
        elif info.get('details', {}).get('path', None) is not None:
            path = info['details']['path']
        else:
            path = self.transform_res_2_path(info)
        
        if filter_key:
            info = {k:v for k, v in info.items() if 'prob' in k}
        fig, ax = plot_matching_result(traj, path, self.net, info, legend=legend)

        return fig, ax


if __name__ == "__main__":
    from .osmnet.build_graph import build_geograph
    net = build_geograph(ckpt = DATA_FOLDER / 'network/Shenzhen_graph_pygeos.ckpt')
    self = ST_Matching(net=net)
    
    traj = self.load_points(DATA_FOLDER / "trajs/traj_0.geojson")
    path, info = self.matching(traj, plot=True, top_k=5)
    
