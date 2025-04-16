"""
@copyright: Dreame Technology(ShenZhen) Co., Ltd.
@author: luchen
@date: 2025-04-15
@description: 
    This code is for self design floor plan method, 
    which can load the floor plan from 2D map or self-designed parameters.
"""
import gin
import shapely
from shapely.geometry import Polygon

import numpy as np
from numpy.random import uniform
from shapely.affinity import translate
from tqdm import tqdm

from .solver import FloorPlanMoves
from .base import room_level, room_type
from .contour import ContourFactory
from .dreame_graph import DreameGraphMaker
from .solidifier import BlueprintSolidifier
from .dreame_segment import DreameSegmentMaker
from infinigen.core.tags import Semantics
from infinigen.core.util.math import FixedSeed

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.constraints.evaluator.evaluate import evaluate_problem


@gin.configurable
class DreameFloorPlanSolver:
    def __init__(self, factory_seed, consgraph, n_divide_trials=100, iters_mult=200):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.constants = consgraph.constants
            self.consgraph = consgraph
            self.n_stories = self.constants.n_stories
            self.contour_factory = ContourFactory(self.consgraph)
            
            self.graphs = []
            self.widths, self.heights = [], []
            self.contours = []

            self.build_graphs() # build the graphs of the floor plan

            self.segment_makers = [
                DreameSegmentMaker(
                    factory_seed,
                    self.constants,
                    consgraph,
                    self.contours[i],
                    self.graphs[i],
                    i,
                )
                for i in range(self.n_stories)
            ]

            self.solver = FloorPlanMoves(self.constants)
            self.solidifiers = [
                BlueprintSolidifier(consgraph, g, i) for i, g in enumerate(self.graphs)
            ]
    
    """
    Initialize the graphs of the floor plan, include the width, height, contours, etc.
    return:
        graphs: the graphs of the floor plan
    """
    def build_graphs(self):
        """
        This function is for building the floor plan graph.
        """
        for i in range(self.n_stories):
            dreame_graph_maker = DreameGraphMaker(self.factory_seed, self.consgraph, i)
            
            graph = dreame_graph_maker.make_graph(np.random.randint(1e6))

            args = (
                [self.widths[-1], self.heights[-1]] 
                if len(self.graphs) > 0 
                else [None, None]
            )

            # width, height = 30.0, 20.0
            width, height = 11.0, 13.0

            # if width is not None and height is not None:
            #     break

            self.graphs.append(graph)
            while len(self.contours) <= i:
                contour = shapely.box(0, 0, width, height)
                if len(self.contours) > 0:
                    x_offset = self.constants.unit_cast(
                        (width - self.widths[0]) * uniform(0, 1)
                    )
                    y_offset = self.constants.unit_cast(
                        (height - self.heights[0]) * uniform(0, 1)
                    )
                    contour = translate(contour, -x_offset, -y_offset)
                    if not self.contours[-1].contains(contour):
                        continue
                self.contours.append(contour)
                # break
            

        self.widths.append(width)
        self.heights.append(height)
    
    """ 
    This the main function for the DreameFloorPlanSolver
    return:
        state: the state of the room
        unique_roomtypes: the unique room types
        dimensions: the dimensions of the room
    """

    def solve(self):

        state = State(graphs=self.graphs)
        states = []
        while len(states) < self.n_stories:
            # pholder_point = [[0, 8], [6, 8], [6, 12], [0, 12]] # initial the pholder for the staircase_candidates in build segments
            pholder_point = [[9, 11], [11, 11], [11, 13], [9, 13]] # initial the pholder for the staircase_candidates in build segments
            pholder = Polygon(pholder_point)
            # pholder = self.contour_factory.add_staircase(self.contours[-1])
            state.objs = {}
            states = []
            for j in range(self.n_stories):
                
                st = self.segment_makers[j].build_segments(pholder)
                if st is not None:
                    states.append(st)
                    state.objs.update(st.objs)
                    break
            else:
                break
        
        """
            change the orignal poisition of the room
        """
        # state = self.simulated_anneal(state)

        """
            decorate the state, like round, triangle, etc.
        """
        # self.contour_factory.decorate(state)

        """
            solidify the state, like add the wall, window, door, etc.
        """
        obj_states = {}
        for j in range(self.n_stories):
            with FixedSeed(self.factory_seed):
                st, room_meshed = self.solidifiers[j].solidify(
                    State({k: v for k, v in state.objs.items() if room_level(k) == j})
                )
            obj_states.update(st.objs)
        unique_roomtypes = set()
        for graph in self.graphs:
            for s in graph.names:
                unique_roomtypes.add(Semantics(room_type(s)))
        dimensions = (
            self.widths[0],
            self.heights[0],
            self.constants.wall_height * self.n_stories,
        )
        return State(obj_states), unique_roomtypes, dimensions


    def simulated_anneal(self, state):
        consgraph = self.consgraph.filter("room")
        consgraph.constraints["graph"] = cl.graph_coherent(self.consgraph.constants)
        score, _ = evaluate_problem(consgraph, state, memo={})
        it = self.iter_per_room * sum(len(g) for g in self.graphs)
        with tqdm(total=it, desc="Sampling solutions") as pbar:
            while pbar.n < it:
                state_ = self.solver.perturb_state(state)
                score_, violated_ = evaluate_problem(consgraph, state_, memo={})
                scale = self.score_scale * pbar.n / it
                if np.log(uniform()) < (score - score_) * scale and not violated_:
                    state = state_
                    score = score_
                pbar.update(1)
                pbar.set_postfix(score=score)
        self.solver.plot(state) 
        return state