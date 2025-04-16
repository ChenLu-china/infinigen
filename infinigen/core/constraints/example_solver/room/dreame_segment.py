"""
@author: luchen
@copyright: Dreame Technology(ShenZhen) Co., Ltd.
@date: 2025-04-15
@description: 2D floor segment maker
"""

import shapely
import shapely.plotting
from shapely import LineString, union, Polygon

import numpy as np
from numpy.random import uniform

from collections import defaultdict
import matplotlib.pyplot as plt

from infinigen.core.tags import Semantics
from infinigen.core.util.math import FixedSeed
from infinigen.core.constraints import constraint_language as cl
from infinigen.assets.utils.shapes import (
    cut_polygon_by_line,
    is_valid_polygon,
    segment_filter,
)

from infinigen.core.constraints.example_solver.state_def import (
    ObjectState,
    RelationState,
    State,
)

from infinigen.core.constraints.constraint_language.constants import RoomConstants


from .base import RoomGraph, room_name, room_type
from .utils import shared


"""
self design the room info, polygon is important
"""
def divide_box_info_gen(graph: RoomGraph):

    info = {}

    names = [room_type(name) for name in graph.names]
    for name in names:
        name_ = room_name(name, 0, 0)
        if name == Semantics.Kitchen:
            info[name_] = {
                    "line": [(22.0, -100), (22.0, 100)], 
                    "polygon": [[22, 0], [30, 0], [30, 20], [22, 20]]
            }
        elif name == Semantics.Bedroom:
            info[name_] = {
                    "line": [[(0, 10), (15, 10)], [(15, 10), (15, 20)]], 
                    "polygon": [[0, 12], [10, 12], [10, 20], [0, 20]]}
        elif name == Semantics.Bathroom:
            info[name_] = {
                    "line": [[(15, 0), (15, 8)], [(0, 8), (15, 8)]],
                    "polygon": [[0, 0], [10, 0], [10, 8], [0, 8]]
            }
        elif name == Semantics.LivingRoom:
            info[name_] = {
                    "line": [(22.0, -100), (22.0, 100)],
                    "polygon": [[10, 0], [22, 0], [22, 20], [10, 20], [10, 12], [0, 12], [0, 8], [10, 8]]
            }
    return info


def normal_room_info_gen(graph: RoomGraph):
    info = {}

    names = [room_type(name) for name in graph.names]
    for name in names:
        name_ = room_name(name, 0, 0)
        if name == Semantics.Bedroom:
            info[name_] = {
                "polygon": [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]]
            }
        elif name == Semantics.Closet:
            info[name_] = {
                "polygon": [[0.0, 4.0], [5.0, 4.0], [5.0, 6.0], [0.0, 6.0]]
            }
        elif name == Semantics.Bathroom:
            info[name_] = {
                "polygon": [[0.0, 6.0], [5.0, 6.0], [5.0, 9.0], [0.0, 9.0]]
            }
        elif name == Semantics.Kitchen:
            info[name_] = {
                "polygon": [[0.0, 9.0], [5.0, 9.0], [5.0, 13.0], [0.0, 13.0]]
            }
        elif name == Semantics.Balcony:
            info[name_] = {
                "polygon": [[5.0, 0.0], [11.0, 0.0], [11.0, 2.5], [5.0, 2.5]]
            }
        elif name == Semantics.LivingRoom:
            info[name_] = {
                "polygon": [[5.0, 2.5], [11.0, 2.5], [11.0, 13.0], [5.0, 13.0]]
            }
    return info



class DreameSegmentMaker:
    def __init__(
        self, 
        factory_seed,
        constants: RoomConstants,
        consgraph, 
        contour,
        graph: RoomGraph, 
        level
    ):
        with FixedSeed(factory_seed):
            self.factory_seed = factory_seed
            self.constants = constants
            self.level = level
            self.contour = contour
            self.consgraph = consgraph
            self.graph = graph

            self.n_boxes = int(len(graph) * uniform(1.8, 2.0))

    """
    This is the main function for building the segments of the floor plan
    """
    def build_segments(self, placeholder=None):
        seed = np.random.randint(10e7)
        while True:
            try:
                with FixedSeed(seed):
                    segments, shared_edges = self.filter_segments() # fliter the segments using pre-defined room info
                break
            except Exception:
                pass
            seed += 1
        
        neighbours_all = {
            k: set(self.constants.filter(se)) for k, se in shared_edges.items()
        }
        """
        Find out the exterior edges and neighbours edges of the segments
        """
        exterior_edges = {}
        exterior_neighbours = []
        for k, s in segments.items():
                    l = s.boundary
                    for ls in shared_edges[k].values():
                        l = l.difference(ls)
                    if l.length > 0:
                        exterior_edges[k] = (
                            shapely.MultiLineString([l]) if isinstance(l, LineString) else l
                        )
                    else:
                        exterior_edges[k] = shapely.MultiLineString([])
                    if segment_filter(l, self.constants.segment_margin):
                        exterior_neighbours.append(k)
        
        """
        Find out the staircase candidates located in the segments
        """
        staircase_candidates = []
        if placeholder is not None:
            for k, s in segments.items():
                if (
                    s.intersection(placeholder).area / placeholder.area
                    > self.constants.staircase_thresh
                ):
                    staircase_candidates.append(k)
            if len(staircase_candidates) == 0:
                return None
            
        # exterior_rooms = self.graph.ns[
        #     self.graph.names.index(room_name(Semantics.Exterior, self.level))
        """
        Build the state of the floor plan
        """
        st = State()
        for i, r in enumerate(self.graph.names):
            if i in self.graph.invalid_indices:
                continue
            st.objs[r] = ObjectState(
                polygon=self.constants.canonicalize(segments[r]),
                relations=[
                    RelationState(cl.SharedEdge(), j, value=se)
                    for j, se in shared_edges[r].items()
                ],
                tags={room_type(r), Semantics.RoomContour},
            )
        
        exterior = room_name(Semantics.Exterior, self.level)
        relations = [
            RelationState(cl.SharedEdge(), j, value=se)
            for j, se in exterior_edges.items()
        ]
        st.objs[exterior] = ObjectState(
            polygon=self.contour,
            relations=relations,
            tags={Semantics.Exterior, Semantics.RoomContour},
        )

        if placeholder is not None:
            pholder = room_name(Semantics.Staircase, self.level)
            st.objs[pholder] = ObjectState(
                polygon=placeholder, tags={Semantics.Staircase}
            )

        return st
    
    def divide_segments(self):

        # gen our room info
        room_info = normal_room_info_gen(self.graph) # gen the room info using pre-defined room info

        segments = {}
        for k in range(len(self.graph.names)):
            print(list(room_info.keys()))
            if self.graph.names[k] in list(room_info.keys()):
                info = room_info[self.graph.names[k]]
                polygon = Polygon(info["polygon"])
                segments[self.graph.names[k]] = polygon
            else:
                continue
        
        return {k: v for k, v in segments.items()}

    def filter_segments(self):
        segments = self.divide_segments()
        
        shared_edges = defaultdict(dict)
        attached = defaultdict(set)
        for k, s in segments.items():
            for l, t in segments.items():
                if k < l:
                    se = shared(s, t)
                    shared_edges[k][l] = shared_edges[l][k] = se
                    if se.length >= self.constants.segment_margin:
                        attached[k].add(l)
                        attached[l].add(k)
        self.plot(segments)
        
        return segments, shared_edges
    
    """
    Visualize the segments
    """
    def plot(self, segments):
        from pathlib import Path
        plt.clf()
        for k, s in segments.items():
            shapely.plotting.plot_polygon(s, color=uniform(0, 1, 3))
        plt.tight_layout()

        # output
        output_dir = Path("./dreame_floor_plan_debug")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"dreame_floorplan_segments_{np.random.randint(0, 10000)}.png"
        plt.savefig(output_dir / filename)
        print(f"Saved to {output_dir / filename}")
        plt.show()