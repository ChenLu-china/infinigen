"""
@author: luchen
@copyright: Dreame Technology(ShenZhen) Co., Ltd.
@date: 2025-04-15
@description: 2D floor graph maker
"""

import operator
import numpy as np
import gin

from infinigen.core.tags import Semantics
from infinigen.core.util.math import FixedSeed
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.constraint_language import Problem

from infinigen.core.constraints.example_solver.room.base import (
    RoomGraph,
    room_name,
    room_type,
)

from infinigen.core.constraints.example_solver.state_def import (
    ObjectState,
    RelationState,
    State,
)

@gin.configurable
class DreameGraphMaker:

    # design the room types we want to generate
    test_room_name = [
                      Semantics.LivingRoom, Semantics.Bathroom, 
                      Semantics.Bedroom, Semantics.Kitchen,
                      Semantics.Closet, Semantics.Balcony
    ]

    def __init__(self, factory_seed, consgraph, level, fast=False):
        self.factory_seed = factory_seed
        with FixedSeed(factory_seed):
            self.data_path = None
            self.level = level
            self.constants = consgraph.constants
            # self.consgraph = consgraph

            consgraph = consgraph.filter("node")
            self.fast = fast
            # self.consgraph = Problem({
            #     "node": consgraph.constraints["node"]
            # }, {
            #     "node_gen": self.inject(consgraph.constraints["node_gen"])
            # }, consgraph.constants)

    @property
    def semantics_floor(self):
        return Semantics.floors[self.level]


    """
    This function is for self-designing the room data, include the room name, room type, room relations.
    return:
        state: the state of the room
    """
    def self_design_room_data(self):
        main_room_name = room_name(Semantics.LivingRoom, self.level, 0)
        state = State({
        })
        for room_type in self.test_room_name:
            name = room_name(room_type, self.level, 0)
            if room_type != Semantics.LivingRoom:
                state[name] = ObjectState(
                    tags={
                        room_type,
                        Semantics.RoomContour,
                        Semantics.Visited,
                        self.semantics_floor,
                    },
                    # add the relations to the other room types
                    relations=[RelationState(cl.Traverse(), main_room_name)]
                )
            else:
                state[name] = ObjectState(
                    tags={
                        room_type,
                        Semantics.RoomContour,
                        Semantics.Visited,
                        self.semantics_floor,
                    },
                    # add the relations to the living room
                    relations=[RelationState(cl.Traverse(), 
                               room_name(self.test_room_name[i], self.level, 0)) for i in range(len(self.test_room_name)) if self.test_room_name[i] != Semantics.LivingRoom]
                )

        entrance = room_name(Semantics.Entrance, self.level, 0) # this is in need for after process
        state[entrance] = ObjectState(
            tags={
                Semantics.Entrance,
                Semantics.RoomContour,
                Semantics.Visited,
                self.semantics_floor,
            },
            relations=[RelationState(cl.Traverse(), main_room_name)]
        )
        
        state[main_room_name].relations.append(RelationState(cl.Traverse(), entrance)) # add the entrance relation to the main room
        
        return state

    # TODO: load the room data from the 2D map
    def load_room_data(self, data_path):
        name = room_name(Semantics.Root, self.level)
        state = State(
            {
                name: ObjectState(
                    tags={
                        Semantics.Root,
                        Semantics.RoomContour,
                        self.semantics_floor,
                        
                    },
                    relations=[],
                )
            }
        )

        if data_path is None:
            state

    """
    The main fuction for the DreameGrapMaker
    return:
        graph: the graph of the room
    """
    def make_graph(self, i):
        with FixedSeed(i):
            if self.data_path is None:
                state = self.self_design_room_data()
            
            graph = self.state2graph(state)
        
        return graph

    """
    Change the state to the graph
    args:
        state: the state of the room
    return:
        graph: the graph of the room
    """
    def state2graph(self, state):
        
        state = self.merge_exterior(state)
        state, entrance = self.merge_entrance(state)

        names = [k for k in state.objs.keys() if room_type(k) != Semantics.Exterior] + [
            room_name(Semantics.Exterior, self.level)
        ]

        return RoomGraph(
            [[names.index(r.target_name) for r in state[n].relations] for n in names],
            names,
            None if entrance is None else names.index(entrance),
        )
    

    def merge_exterior(self, state):
        exterior_connected = set()
        for k, obj_st in state.objs.items():
            if room_type(k) == Semantics.Exterior:
                for r in obj_st.relations:
                    exterior_connected.add(r.target_name)
        exterior_name = room_name(Semantics.Exterior, self.level)
        state = State(
            {
                k: obj_st
                for k, obj_st in state.objs.items()
                if room_type(k) != Semantics.Exterior
            }
        )
        for k in exterior_connected:
            state[k].relations = [
                r
                for r in state[k].relations
                if room_type(r.target_name) != Semantics.Exterior
            ]
            state[k].relations.append(RelationState(cl.Traverse(), exterior_name))
        state[exterior_name] = ObjectState(
            tags={Semantics.Exterior, Semantics.RoomContour, self.semantics_floor},
            relations=[RelationState(cl.Traverse(), k) for k in exterior_connected],
        )
        return state

    __call__ = make_graph

    def merge_entrance(self, state):
        entrance_connected = set()
        for k, obj_st in state.objs.items():
            if room_type(k) == Semantics.Entrance:
                for r in obj_st.relations:
                    entrance_connected.add(r.target_name)
        state = State(
            {
                k: obj_st
                for k, obj_st in state.objs.items()
                if room_type(k) != Semantics.Entrance
            }
        )
        for k in entrance_connected:
            state[k].relations = [
                r
                for r in state[k].relations
                if room_type(r.target_name) != Semantics.Entrance
            ]
        if len(entrance_connected) == 0:
            entrance = None
        else:
            entrance = np.random.choice(list(entrance_connected))
            exterior_name = room_name(Semantics.Exterior, self.level)
            state[entrance].relations.append(
                RelationState(cl.Traverse(), exterior_name)
            )
            if exterior_name not in state.objs:
                state[exterior_name] = ObjectState(
                    tags={
                        Semantics.Exterior,
                        Semantics.RoomContour,
                        self.semantics_floor,
                    }
                )
            state[exterior_name].relations.append(
                RelationState(cl.Traverse(), entrance)
            )
        return state, entrance


