import os
os.environ["USE_PYGEOS"] = "1"

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from .graph import GeoDigraph
from .geo.metric import lcss, edr, erp
from .geo.douglasPeucker import simplify_trajetory_points

from .osmnet.build_graph import build_geograph

from .match.status import STATUS
from .match.io import load_points
from .match.postprocess import get_path
from .match.candidatesGraph import construct_graph
from .match.spatialAnalysis import analyse_spatial_info
from .match.geometricAnalysis import analyse_geometric_info
from .match.projection import project_traj_points_to_network
from .match.viterbi import process_viterbi_pipeline, find_matched_sequence
from .match.visualization import matching_debug_level, plot_matching_result

from .utils.timer import timeit
from .utils.logger_helper import make_logger
from .utils.misc import SET_PANDAS_LOG_FORMET
from .setting import DATA_FOLDER, DEBUG_FOLDER, DIS_FACTOR

SET_PANDAS_LOG_FORMET()


class ST_Matching():
    def __init__(self,
                 net: GeoDigraph,
                 dp_thres=5,
                 max_search_steps=2000,
                 max_search_dist=10000,
                 top_k_candidates=5,
                 cand_search_radius=50,
                 crs_wgs=4326,
                 crs_prj=900913,
                 prob_thres=.8
                 ):
        self.net = net
        self.dp_thres = dp_thres
        self.crs_wgs = crs_wgs
        self.crs_wgs = crs_wgs
        self.dis_factor = DIS_FACTOR
        self.debug_folder = DEBUG_FOLDER
        self.logger = make_logger('../log', console=False, level="INFO")
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

        # hyper parameters
        self.prob_thres = prob_thres
        self.top_k_candidates = top_k_candidates
        self.cand_search_radius = cand_search_radius
        self.route_planning_max_search_steps = max_search_steps
        self.route_planning_max_search_dist = max_search_dist

    @timeit
    def matching(self, traj, top_k=None, dir_trans=False, beam_search=True,
                 simplify=True, tolerance=10, plot=False, save_fn=None,
                 debug_in_levels=False, details=False):
        res = {'status': STATUS.UNKNOWN}
        
        # simplify trajectory: (tolerance, 10 meters)
        if simplify:
            ori_traj = traj
            traj = traj.copy()
            traj = self._simplify(traj, tolerance=tolerance)

        # geometric analysis
        top_k = top_k if top_k is not None else self.top_k_candidates
        cands = analyse_geometric_info(
            points=traj, edges=self.net.df_edges, top_k=top_k, radius=self.cand_search_radius)
        
        # is_valid
        s, route = self._is_valid(traj, cands, res)
        if not s:
            return res

        # spatial analysis
        res['probs'] = {}
        rList, graph = self._spatial_analysis(traj, cands, dir_trans, beam_search, metric=res['probs'])
        match_res, steps = get_path(self.net, traj, rList, graph, cands, metric=res['probs'])
        res.update(match_res)

        if details:
            _dict = {'cands': cands,'rList': rList, 'graph': graph, 'route': route, "steps": steps}
            res['details'] = _dict

        if plot or save_fn:
            fig, ax = plot_matching_result(traj, route, self.net)
            if simplify:
                ori_traj.plot(ax=ax, color='gray', alpha=.3)
                traj.plot(ax=ax, color='yellow', alpha=.5)
            if not plot:
               plt.close() 
            if save_fn:
                fig.savefig(save_fn, dpi=300, bbox_inches='tight', pad_inches=0.02)

        if debug_in_levels:
            self.matching_debug(traj, graph)
        
        return res

    def _is_valid(self, traj, cands, info, eps = 1e-7):
        # -> status, route
        if cands is None:
            info['status'] = STATUS.NO_CANDIDATES
            return False, None
        
        # Only one single point matched
        if traj.shape[0] == 1 or cands.pid.nunique() == 1: 
            eid = cands.sort_values('dist_p2c').head(1).eid.values
            coord = cands.iloc[0].projection
            res = {'eids': eid, 'step_0': [coord, [coord[0] + eps, coord[1] + eps]]}
            info.update(res)
            info['status'] = STATUS.ONE_POINT
            
            return False, res
        
        return True, None

    def _spatial_analysis(self, traj, cands, dir_trans, beam_search, metric={}):
        if not beam_search:
            graph = analyse_spatial_info(self.net, traj, cands, dir_trans)
            prob, rList = process_viterbi_pipeline(cands, graph[['pid_1', 'dist_prob']])
        else:
            graph = construct_graph(traj, cands, dir_trans=dir_trans)
            prob, rList, graph = find_matched_sequence(cands, graph, self.net, dir_trans)

        metric['prob'] = prob

        return rList, graph

    def eval(self, traj, path, eps=10, metric='lcss', g=None):
        assert metric in ['lcss', 'edr', 'erp']
        path_points = np.concatenate(path.geometry.apply(lambda x: x.coords[:]).values)
        traj_points = np.concatenate(traj.geometry.apply(lambda x: x.coords[:]).values)
        
        # FIXME 是否使用 轨迹节点 和 投影节点 作比较
        # projected_points = info['rList'][['pid', 'eid']].merge(info['cands'], on=['pid', 'eid'])
        # points = np.concatenate(projected_points.point_geom.apply(lambda x: x.coords[:]).values)
        # projections = np.concatenate(projected_points.projection.apply(lambda x: x).values).reshape((-1, 2))

        eval_funs = {
            'lcss': [lcss, (traj_points, path_points, eps)], 
            'edr': [edr, (traj_points, path_points, eps)], 
            'edp': [erp, (traj_points, path_points, g)]
        }
        _eval = eval_funs[metric]

        return _eval[0](*_eval[1])

    def project(self, traj_panos, path, keep_attrs=None):
        return project_traj_points_to_network(traj_panos, path, self.net, keep_attrs)

    def load_points(self, fn, compress=False, dp_thres: int = None,
                    crs: int = None, in_sys: str = 'wgs', out_sys: str = 'wgs'):
        
        traj, _ = load_points(fn, compress, dp_thres, crs, in_sys, out_sys)
        
        return traj

    def _simplify(self, points:gpd.GeoDataFrame, tolerance:int=None, inplace=False):        
        return simplify_trajetory_points(points, tolerance, inplace=True, logger=self.logger)

    def matching_debug(self, traj, graph, debug_folder='./debug'):
        """matching debug

        Args:
            traj ([type]): Trajectory
            tList ([type]): [description]
            graph_t ([type]): [description]
            net ([Digraph_OSM]): [description]
            debug (bool, optional): [description]. Defaults to True.
        """
        graph = gpd.GeoDataFrame(graph)
        graph.geometry = graph.whole_path

        layer_ids = graph.index.get_level_values(0).unique().sort_values().values
        for layer in layer_ids:
            df_layer = graph.loc[layer]
            matching_debug_level(self.net, traj, df_layer, layer, debug_folder)
        
        return

    def plot_result(self, traj, info):
        path = self.transform_res_2_path(info)
        fig, ax = plot_matching_result(traj, path, self.net)
        if not info:
            return fig, ax

        text = []
        for key, val in info['probs'].items():
            if isinstance(val, float):
                _str = f"{key}: {val * 100: .2f} %"
            else:
                _str = f"{key}: {val}"
            text.append(_str)

        x0, x1, y0, y1 = ax.axis()
        ax.text(x0 + (x1- x0)/50, y0 + (y1 - y0)/50, "\n".join(text))

        return fig, ax

    def transform_res_2_path(self, res):
        path = self.net.get_edge(res['eids'], reset_index=True)
        path.loc[0, 'geometry'] = LineString(res['step_0'])
        if 'step_n' in res:
            n = path.shape[0] - 1
            path.loc[n, 'geometry'] = LineString(res['step_n'])
        
        path = path[~path.geometry.is_empty]

        return path

    def get_points(self, traj, ids):
        return NotImplementedError
    
    def points_to_polyline(self, points):
        return NotImplementedError

    def polyline_to_points(self, polyline):
        return NotImplementedError


if __name__ == "__main__":
    net = build_geograph(ckpt = DATA_FOLDER / 'network/Shenzhen_graph_pygeos.ckpt')
    self = ST_Matching(net=net)
    
    traj = self.load_points(DATA_FOLDER / "trajs/traj_0.geojson")
    path, info = self.matching(traj, plot=True, top_k=5)
    
