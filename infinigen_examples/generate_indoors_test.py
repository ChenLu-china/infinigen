import bpy
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import gin
import numpy as np

from infinigen import repo_root
from infinigen.core import tags as t
from infinigen.core.constraints import checks
from infinigen.core import init, execute_tasks, placement
from infinigen.core.util import pipeline
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.example_solver import (greedy, state_def, populate)

from infinigen_examples.constraints import home as home_constraints
from infinigen.core.constraints.example_solver.solve import Solver
from infinigen.core.constraints.example_solver.room import decorate as room_dec
from infinigen.core.util import blender as butil

from infinigen.core.util.test_utils import (
    import_item,
    load_txt_list,
)
from infinigen.assets.placement.floating_objects import FloatingObjectPlacement


from .constraints import util as cu
from .util.generate_indoors_util import (
    restrict_solving,
    apply_greedy_restriction,
)
logger = logging.getLogger(__name__)


def default_greedy_stages():

    on_floor = cl.StableAgainst({}, cu.floortags)
    on_wall = cl.StableAgainst({}, cu.walltags)
    on_ceiling = cl.StableAgainst({}, cu.ceilingtags)
    side = cl.StableAgainst({}, cu.side)

    all_room = r.Domain({t.Semantics.Room, -t.Semantics.Object})

    all_obj = r.Domain({t.Semantics.Object, -t.Semantics.Room})

    all_obj_in_room = all_obj.with_relation(
        cl.AnyRelation(), all_room.with_tags(cu.variable_room)
    )
    primary = all_obj_in_room.with_relation(-cl.AnyRelation(), all_obj)

    greedy_stages = {}

    greedy_stages["rooms"] = all_room

    greedy_stages["on_floor_and_wall"] = primary.with_relation(
        on_floor, all_room
    ).with_relation(on_wall, all_room)

    greedy_stages["on_floor_freestanding"] = primary.with_relation(
        on_floor, all_room
    ).with_relation(-on_wall, all_room)

    greedy_stages["on_wall"] = (
        primary.with_relation(-on_floor, all_room)
        .with_relation(-on_ceiling, all_room)
        .with_relation(on_wall, all_room)
    )
    greedy_stages["on_ceiling"] = (
        primary.with_relation(-on_floor, all_room)
        .with_relation(on_ceiling, all_room)
        .with_relation(-on_wall, all_room)
    )

    secondary = all_obj.with_relation(
        cl.AnyRelation(), all_obj_in_room.with_tags(cu.variable_obj)
    )

    greedy_stages["side_obj"] = (
        secondary.with_relation(side, all_obj)
        .with_relation(-cu.on, all_obj)
        .with_relation(-cu.ontop, all_obj)
    )

    greedy_stages["obj_ontop_obj"] = (
        secondary.with_relation(-side, all_obj)
        .with_relation(cu.ontop, all_obj)
        .with_relation(-cu.on, all_obj)
    )
    greedy_stages["obj_on_support"] = (
        secondary.with_relation(-side, all_obj)
        .with_relation(cu.on, all_obj)
        .with_relation(-cu.ontop, all_obj)
    )
    # greedy_stages["on_wall"] = ()

    # greedy_stages["on_floor_and_wall"] = primary.with_relation(
    #     on_floor, all_room).with_relation(on_wall, all_room)
    
    return greedy_stages


all_vars = [cu.variable_room, cu.variable_obj]

@gin.configurable
def compose_indoors(output_folder: Path, scene_seed: int, **overrides):

    p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)

    # logger.debug(overrides)

    consgraph = home_constraints.home_furniture_constraints()
    consgraph_rooms = home_constraints.home_room_constraints() # indoor ruls init essination
    constants = consgraph_rooms.constants
    stages = default_greedy_stages()
    # checks.check_all(consgraph, stages, all_vars)

    # TODO: remove this
    stages, consgraph, limits = restrict_solving(stages, consgraph)
    
    if overrides.get("restrict_single_supported_roomtype", False):
        restrict_parent_rooms = {
            np.random.choice(
                [
                    # Only these roomtypes have constraints written in home_furniture_constraints.
                    # Others will be empty-ish besides maybe storage and plants
                    # TODO: add constraints to home_furniture_constraints for garages, offices, balconies, etc
                    t.Semantics.Bedroom,
                    t.Semantics.LivingRoom,
                    t.Semantics.Kitchen,
                    t.Semantics.Bathroom,
                    t.Semantics.DiningRoom,
                ]
            )
        }
        logger.info(f"Restricting to {restrict_parent_rooms}")
        apply_greedy_restriction(stages, restrict_parent_rooms, cu.variable_room)
    
    solver = Solver(output_folder=output_folder)

    def solve_rooms():
        return solver.solve_rooms(scene_seed, consgraph_rooms, stages["rooms"])
    
    state: state_def.State = p.run_stage("solve_rooms", solve_rooms, use_chance=False)

    camera_rigs = placement.camera.spawn_camera_rigs()

    # chair_filter = r.Domain({t.Semantics.Chair}, [(cl.AnyRelation(), stages["rooms"])])
    # p.run_stage(
    #     "abo_chairs", 
    #     lambda: room_dec.test_abo_decorate(solver.get_bpy_objects(chair_filter), constants),
    #     use_chance=False,
    # )

    door_filter = r.Domain({t.Semantics.Door}, [(cl.AnyRelation(), stages["rooms"])])
    window_filter = r.Domain({t.Semantics.Window}, [(cl.AnyRelation(), stages["rooms"])])
    
    p.run_stage(
        "room_doors", 
        lambda: room_dec.populate_doors(solver.get_bpy_objects(door_filter), constants), 
        use_chance=False,
    )
    

    p.run_stage(
        "room_windows",
        lambda: room_dec.populate_windows(solver.get_bpy_objects(window_filter), constants, state),
        use_chance=False,
    )

    def solve_stage_name(stage_name: str, group: str, **kwargs):
        assigments = greedy.iterate_assignments(
            stages[stage_name], state, all_vars, limits
        )

        for i, vars in enumerate(assigments):
            solver.solve_objects(
                consgraph,
                stages[stage_name],
                vars,
                n_steps=overrides[f"solve_steps_{group}"],
                desc=f"{stage_name}_{i}",
                abort_unsatisfied=overrides.get(f"abort_unsatisfied_{group}", False),
                **kwargs,
            )
    
    def solve_large():
        solve_stage_name("on_floor_and_wall", "large")
        solve_stage_name("on_floor_freestanding", "large")

    p.run_stage("solve_large", solve_large, use_chance=False, default=state)

    solved_rooms = [
        state.objs[assignment[cu.variable_room]].obj
        for assignment in greedy.iterate_assignments(
            stages["on_floor_freestanding"], state, [cu.variable_room], limits
        )
    ]
    solved_bound_points = np.concatenate([butil.bounds(r) for r in solved_rooms])
    solved_bbox = (
        np.min(solved_bound_points, axis=0),
        np.max(solved_bound_points, axis=0),
    )

    house_bbox = np.concatenate(
        [
            butil.bounds(obj)
            for obj in solver.get_bpy_objects(r.Domain({t.Semantics.Room}))
        ]
    )
    house_bbox = (
        np.min(house_bbox, axis=0),
        np.max(house_bbox, axis=0),
    )

    p.run_stage(
        "populate_intermediate_pholders",
        populate.populate_state_placeholders,
        solver.state,
        filter=t.Semantics.AssetPlaceholderForChildren,
        final=False,
        use_chance=False,
    )

    # def solve_medium():
    #     solve_stage_name("on_wall", "medium")
    #     solve_stage_name("on_ceiling", "medium")
    #     solve_stage_name("side_obj", "medium")

    # p.run_stage("solve_medium", solve_medium, use_chance=False, default=state)

    # def solve_small():
    #     solve_stage_name("obj_ontop_obj", "small", addition_weight_scalar=3)
    #     solve_stage_name("obj_on_support", "small", restrict_moves=["addition"])

    # p.run_stage("solve_small", solve_small, use_chance=False, default=state)

    solver.optim.save_stats(output_folder / "optim_records.csv")

    p.run_stage(
        "populate_assets", populate.populate_state_placeholders, state, use_chance=False
    )

    def place_floating():
        pholder_rooms = butil.get_collection("placeholders:room_meshes")
        pholder_cutters = butil.get_collection("placeholders:portal_cutters")
        pholder_objs = butil.get_collection("placeholders")

        obj_fac_names = load_txt_list(
            repo_root() / "tests" / "assets" / "list_indoor_meshes.txt"
        )
        facs = [import_item(path) for path in obj_fac_names]

        placer = FloatingObjectPlacement(
            generators=facs,
            camera=camera_rigs[0].children[0],
            background_objs=list(pholder_cutters.objects) + list(pholder_rooms.objects),
            collision_objs=list(pholder_objs.objects),
        )

        placer.place_objs(
            num_objs=overrides.get("num_floating", 20),
            normalize=overrides.get("norm_floating_size", True),
            collision_placed=overrides.get("enable_collision_floating", False),
            collision_existing=overrides.get("enable_collision_solved", False),
        )

    p.run_stage("floating_objs", place_floating, use_chance=False, default=state)


    room_meshed = butil.get_collection("placeholders:room_meshes")
    room_split = room_dec.split_rooms(list(room_meshed.objects))

    p.run_stage("room_pillars", room_dec.room_pillars, room_split["wall"].objects, constants)

    p.run_stage("room_walls", room_dec.room_walls, room_split["wall"].objects, constants, use_chance=False)

    p.run_stage("room_floors", room_dec.room_floors, room_split["floor"].objects, use_chance=False)

    p.run_stage("room_ceilings", room_dec.room_ceilings, room_split["ceiling"].objects, use_chance=False)

    bpy.ops.wm.save_as_mainfile(filepath=str(output_folder / "floorplan.blend"))
    logger.info(f"Saved floorplan to {output_folder / 'floorplan.blend'}")

    return {
        "height_offset": None,
        "whole_bbox": None,
    }

def main(args):
    scene_seed = init.apply_scene_seed(args.seed)

    init.apply_gin_configs(
        configs=["base_indoors.gin"] + args.configs,
        overrides=args.overrides,
        config_folders=[
            "infinigen_examples/configs_indoor",
            "infinigen_examples/configs_nature",
        ],
    )

    execute_tasks.main(
        compose_scene_func=compose_indoors,
        populate_scene_func=None,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        task=args.task,
        task_uniqname=args.task_uniqname,
        scene_seed=scene_seed,
    )
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--input_folder", type=Path, default=None)
    parser.add_argument("-s", 
        "--seed", 
        default=None, 
        help="The seed used to generate the scene"
    )

    parser.add_argument("-t", 
            "--task",
            nargs="+",
            default=["coarse"],
            choices=[
                "coarse",
                "populate",
                "fine_terrain",
                "ground_truth",
                "render",
                "mesh_save",
                "export",
            ],
    )
    parser.add_argument(
        "-g",
        "--configs",
        nargs="+",
        default=["base"],
        help="Set of config files for gin (separated by spaces) "
        "e.g. --gin_config file1 file2 (exclude .gin from path)",
    )
    parser.add_argument(
        "-p",
        "--overrides",
        nargs="+",
        default=[],
        help="Parameter settings that override config defaults "
        "e.g. --gin_param module_1.a=2 module_2.b=3",
    )
    parser.add_argument("--task_uniqname", type=str, default=None)
    parser.add_argument("-d", "--debug", type=str, nargs="*", default=None)
  
    args = init.parse_args_blender(parser)

    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    if args.debug is not None:
        for name in logging.root.manager.loggerDict:
            if not name.startswith("infinigen"):
                continue
            if len(args.debug) == 0 or any(name.endswith(x) for x in args.debug):
                logging.getLogger(name).setLevel(logging.DEBUG)

    main(args)