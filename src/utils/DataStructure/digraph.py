import heapq
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import deque
from haversine import haversine, Unit


class Node:
    """
    Define the node in the road network 
    """

    def __init__(self, id):
        self.val = id
        self.x, self.y = [float(i) for i in id.split(',')]
        self.prev = set()
        self.nxt = set()
        self.indegree = 0
        self.outdegree = 0

    def add(self, point):
        self.nxt.add(point)
        self.outdegree += 1

        point.prev.add(self)
        point.indegree += 1

    def check_0_out_more_2_in(self):
        return self.outdegree == 0 and self.indegree >= 2

    def move_nxt_to_prev(self, node):
        if node not in self.nxt:
            return False

        self.nxt.remove(node)
        self.prev.add(node)
        self.indegree += 1
        self.outdegree -= 1
        return True

    def move_prev_to_nxt(self, node):
        if node not in self.prev:
            return False

        self.prev.remove(node)
        self.nxt.add(node)
        self.indegree -= 1
        self.outdegree += 1
        return True


class Digraph:
    def __init__(self, edges:list=None, nodes:dict=None, *args, **kwargs):
        """[summary]

        Args:
            edges (list, optional): Shape: (N, 2/3). Defaults to None.
            nodes (dict, optional): [description]. Defaults to None.
        """
        self.graph = {}
        self.graph_r = {}
        self.edge  = {}
        self.node  = {}
        
        if edges is not None:
            self.build_graph(edges)

        if nodes is not None:
            assert isinstance(nodes, dict), "Check the Node format"
            self.node = nodes
        
        self.calculate_degree()


    def __str__(self):
        return ""


    def add_edge(self, start, end, length=None):
        for p in [start, end]:
            for g in [self.graph, self.graph_r]:
                if p in g:
                    continue
                g[p] = set()
            
        self.graph[start].add(end)
        self.graph_r[end].add(start)
        if length is not None:
            self.edge[(start, end)] = length
            
        pass


    def remove_edge(self, start, end):
        self.graph[start].remove(end)
        if len(self.graph[start]) == 0:
            del self.graph[start]
        
        self.graph_r[end].remove(start)
        if len(self.graph_r[end]) == 0:
            del self.graph_r[end]
        pass


    def build_graph(self, edges):
        for edge in edges:
            start, end, length = edge
            assert not(np.isnan(start) or np.isnan(end)), f"Check the input ({start}, {end})"
            
            if isinstance(start, float):
                start = int(start)
            if isinstance(end, float):
                end = int(end)
            
            self.add_edge(start, end, length)
        
        return self.graph


    def clean_empty_set(self):
        for item in [self.graph_r, self.graph]:
            for i in list(item.keys()):
                if len(item[i]) == 0:
                    del item[i]
        pass


    def calculate_degree(self,):
        self.clean_empty_set()
        self.degree = pd.merge(
            pd.DataFrame([[key, len(self.graph_r[key])]
                          for key in self.graph_r], columns=['pid', 'indegree']),
            pd.DataFrame([[key, len(self.graph[key])]
                          for key in self.graph], columns=['pid', 'outdegree']),
            how='outer',
            on='pid'
        ).fillna(0).astype(np.int).set_index('pid')
        
        return self.degree


    def get_origin_point(self,):
        
        return self.calculate_degree().reset_index().query( "indegree == 0 and outdegree != 0" ).pid.values


    def cal_nodes_dist(self, src, dst):
        return NotImplementedError

    
    def _simpify(self):
        """
        Simplify the graph, namely combine the edges with 1 indegree and 1 out degree
        """
        return NotImplementedError


    def search(self, src, dst, *args, **kwargs):
        return NotImplementedError
    

    def _get_aux_nodes(self, exclude_list=None):
        if getattr(self, 'degree') is None:
            self.calculate_degree()
        
        aux_nids = self.degree.query( "indegree == 1 and outdegree == 1" ).index.unique()
        if exclude_list is not None:
            aux_nids = [id for id in aux_nids if id not in exclude_list]

        return aux_nids


class DigraphAstar(Digraph):
    def __init__(self, edges=None, nodes=None, *args, **kwargs):
        """_summary_

        Args:
            edges (_type_, optional): _description_. Defaults to None. Attrs: [src, dst, dist]
            nodes (_type_, optional): _description_. Defaults to None. Attrs: [x, y]
        """
        super().__init__(edges, nodes, *args, **kwargs)
        
        self.route_planning_memo = {}
        self.node_pair_dis_memo = {}


    def a_star(self, src, dst, max_steps=2000, max_dist=10000):
        """Route planning by A star algs

        Args:
            origin ([type]): [description]
            dest ([type]): [description]
            verbose (bool, optional): [description]. Defaults to False.
            plot (bool, optional): [description]. Defaults to False.

        Returns:
            dict: The route planning result with path, cost and status.
            status_dict = {-1: 'unreachable'}
        """
        if (src, dst) in self.route_planning_memo:
            res = self.route_planning_memo[(src, dst)]
            return res
        
        if src not in self.graph or dst not in self.graph:
            print(f"Edge({src}, {dst})",
                f"{', origin not in graph' if src not in self.graph else ', '}",
                f"{', dest not in graph' if dst not in self.graph else ''}")
            
            return None

        queue = [(0, src)]
        came_from = {src: None}
        distance = {src: 0}
        search_steps = 0

        while queue:
            _, cur = heapq.heappop(queue)
            if cur == dst or search_steps > max_steps:
                break
            
            for nxt in self.graph[cur]:
                if nxt not in self.graph:
                    continue
                
                new_cost = distance[cur] + self.edge[(cur, nxt)]
                if nxt not in distance or new_cost < distance[nxt]:
                    distance[nxt] = new_cost
                    if distance[nxt] > max_dist:
                        continue

                    heapq.heappush(queue, (new_cost + self.cal_nodes_dist(dst, nxt), nxt) )
                    came_from[nxt] = cur
            search_steps += 1

        # abnormal situation
        if cur != dst:
            res = {'path': None, 'cost': np.inf, "status": -1} 
            self.route_planning_memo[(src, dst)] = res
            return res

        # reconstruct the route
        route, queue = [dst], deque([dst])
        while queue:
            node = queue.popleft()
            if came_from[node] is None:
                continue
            route.append(came_from[node])
            queue.append(came_from[node])
        route = route[::-1]

        res = {'path':route, 'cost': distance[dst], 'status':1}
        self.route_planning_memo[(src, dst)] = res

        return res


    def cal_nodes_dist(self, src, dst):
        assert src in self.node and dst in self.node, "Check the input o and d."
        if (src, dst) in self.node_pair_dis_memo:
            return self.node_pair_dis_memo[(src, dst)]
        
        return haversine(
            (self.node[src]['y'], self.node[src]['x']), 
            (self.node[dst]['y'], self.node[dst]['x']), 
            unit=Unit.METERS
        )

