# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import argparse
import logging
from pathlib import Path
import mathutils
# ruff: noqa: E402
# NOTE: logging config has to be before imports that use logging
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import bpy
import gin
import numpy as np
from infinigen import repo_root
#from infinigen.assets import lighting
from infinigen.assets.materials import invisible_to_camera
from infinigen.assets.objects.wall_decorations.skirting_board import make_skirting_board
from infinigen.assets.placement.floating_objects import FloatingObjectPlacement
from infinigen.assets.utils.decorate import read_co
from infinigen.core import execute_tasks, init, placement, surface, tagging
from infinigen.core import tags as t
from infinigen.core.constraints import checks
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.example_solver import (
    greedy,
    populate,
    state_def,
)
from infinigen.core.constraints.example_solver.room import decorate as room_dec
from infinigen.core.constraints.example_solver.solve import Solver
from infinigen.core.placement import camera as cam_util
from infinigen.core.util import blender as butil
from infinigen.core.util import pipeline
from infinigen.core.util.camera import points_inview
from infinigen.core.util.imu import save_imu_tum_files
from infinigen.core.util.test_utils import (
    import_item,
    load_txt_list,
)
from infinigen.terrain import Terrain
from infinigen_examples.constraints import home as home_constraints

from . import (
    generate_nature,  # noqa F401 # needed for nature gin configs to load  # noqa F401 # needed for nature gin configs to load
)
from .constraints import util as cu
from .util.generate_indoors_util import (
    apply_greedy_restriction,
    create_outdoor_backdrop,
    hide_other_rooms,
    place_cam_overhead,
    restrict_solving,
)
# import bmesh 

logger = logging.getLogger(__name__)
#ROOMTYPE = "break"
#ROOMTYPE = "bedroom"
ROOMTYPE = "bathroom"
#ROOMTYPE = "living"
#ROOMTYPE = "dining"
def default_greedy_stages():
    """Returns descriptions of what will be covered by each greedy stage of the solver.

    Any domain containing one or more VariableTags is greedy: it produces many separate domains,
        one for each possible assignment of the unresolved variables.
    """

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

    return greedy_stages


all_vars = [cu.variable_room, cu.variable_obj]


@gin.configurable
def compose_indoors(output_folder: Path, scene_seed: int, **overrides):
    p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)

    logger.debug(overrides)

    def add_coarse_terrain():
        terrain = Terrain(
            scene_seed,
            surface.registry,
            task="coarse",
            on_the_fly_asset_folder=output_folder / "assets",
        )
        terrain_mesh = terrain.coarse_terrain()
        # placement.density.set_tag_dict(terrain.tag_dict)
        return terrain, terrain_mesh

    terrain, terrain_mesh = p.run_stage(
        "terrain", add_coarse_terrain, use_chance=False, default=(None, None)
    )

    #p.run_stage("sky_lighting", lighting.sky_lighting.add_lighting, use_chance=False)

    consgraph = home_constraints.home_furniture_constraints()
    consgraph_rooms = home_constraints.home_room_constraints()
    constants = consgraph_rooms.constants

    stages = default_greedy_stages()
    checks.check_all(consgraph, stages, all_vars)

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
    house_bbox = (np.min(house_bbox, axis=0), np.max(house_bbox, axis=0))

    camera_rigs = placement.camera.spawn_camera_rigs()

    def pose_cameras():
        nonroom_objs = [
            o.obj for o in state.objs.values() if t.Semantics.Room not in o.tags
        ]
        scene_objs = solved_rooms + nonroom_objs

        scene_preprocessed = placement.camera.camera_selection_preprocessing(
            terrain=None, scene_objs=scene_objs
        )

        solved_floor_surface = butil.join_objects(
            [
                tagging.extract_tagged_faces(o, {t.Subpart.SupportSurface})
                for o in solved_rooms
            ]
        )

        placement.camera.configure_cameras(
            camera_rigs,
            scene_preprocessed=scene_preprocessed,
            init_surfaces=solved_floor_surface,
            nonroom_objs=nonroom_objs,
            terrain_coverage_range=None,  # do not filter cameras by terrain visibility, even if nature scenetype configs request this
        )
        butil.delete(solved_floor_surface)
        return scene_preprocessed

    #scene_preprocessed = p.run_stage("pose_cameras", pose_cameras, use_chance=False)

    # def animate_cameras():
    #     cam_util.animate_cameras(
    #         camera_rigs,
    #         solved_bbox,
    #         scene_preprocessed,
    #         pois=[],
    #         terrain_coverage_range=None,  # same as above - do not filter by terrain visiblity when indoors
    #     )

    #     frames_folder = output_folder.parent / "frames"
    #     animated_cams = [cam for cam in camera_rigs if cam.animation_data is not None]
    #     save_imu_tum_files(frames_folder / "imu_tum", animated_cams)

    # p.run_stage(
    #     "animate_cameras", animate_cameras, use_chance=False, prereq="pose_cameras"
    # )

    p.run_stage(
        "populate_intermediate_pholders",
        populate.populate_state_placeholders,
        solver.state,
        filter=t.Semantics.AssetPlaceholderForChildren,
        final=False,
        use_chance=False,
    )

    def solve_medium():
        solve_stage_name("on_wall", "medium")
        solve_stage_name("on_ceiling", "medium")
        solve_stage_name("side_obj", "medium")

    p.run_stage("solve_medium", solve_medium, use_chance=False, default=state)

    def solve_small():
        solve_stage_name("obj_ontop_obj", "small", addition_weight_scalar=3)
        solve_stage_name("obj_on_support", "small", restrict_moves=["addition"])

    p.run_stage("solve_small", solve_small, use_chance=False, default=state)

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

    door_filter = r.Domain({t.Semantics.Door}, [(cl.AnyRelation(), stages["rooms"])])
    window_filter = r.Domain(
        {t.Semantics.Window}, [(cl.AnyRelation(), stages["rooms"])]
    )
    p.run_stage(
        "room_doors",
        lambda: room_dec.populate_doors(solver.get_bpy_objects(door_filter), constants),
        use_chance=False,
    )
    # p.run_stage(
    #     "room_windows",
    #     lambda: room_dec.populate_windows(
    #         solver.get_bpy_objects(window_filter), constants, state
    #     ),
    #     use_chance=False,
    # )

    room_meshes = solver.get_bpy_objects(r.Domain({t.Semantics.Room}))
    p.run_stage(
        "room_stairs",
        lambda: room_dec.room_stairs(constants, state, room_meshes),
        use_chance=False,
    )
    # p.run_stage(
    #     "skirting_floor",
    #     lambda: make_skirting_board(constants, room_meshes, t.Subpart.SupportSurface),
    # )
    # p.run_stage(
    #     "skirting_ceiling",
    #     lambda: make_skirting_board(constants, room_meshes, t.Subpart.Ceiling),
    # )

    rooms_meshed = butil.get_collection("placeholders:room_meshes")

    
    rooms_split = room_dec.split_rooms(list(rooms_meshed.objects))
    print("ROOM SPLIT", rooms_split.keys())
    
    def safe_delete_object(obj):
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        if obj.data is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Remove exterior and ceiling objects
    for category in ['exterior', 'ceiling']:
        if category in rooms_split:
            for obj in list(rooms_split[category].objects):
                safe_delete_object(obj)

    # Filter wall and floor to only include "living" meshes
    living_room_objects = []
    for category in ["wall", "floor"]:
        if category in rooms_split:
            for obj in list(rooms_split[category].objects):
                if ROOMTYPE not in obj.name.lower():
                    safe_delete_object(obj)
                else:
                    living_room_objects.append(obj)

    # Remove doors that aren't associated with living room
    door_collection = bpy.data.collections.get('unique_assets:doors')
    if door_collection:
        # Calculate bounds of living room to check door proximity
        if living_room_objects:
            living_bounds = np.array([butil.bounds(obj) for obj in living_room_objects])
            min_bound = np.min(living_bounds[:, 0], axis=0)
            max_bound = np.max(living_bounds[:, 1], axis=0)
            center = (min_bound + max_bound) / 2

            #print(state.objs)
            # Remove doors that are not associated with living room
            for door in list(door_collection.objects):
                keep_door = False
                
                # Check door relationships in state
               
                for relation in state.objs[door.parent.name].relations:

                    #print(door.parent.name,relation)
                    if ROOMTYPE in relation.target_name :
                        keep_door = True
                        break
                
                
                if not keep_door:
                    safe_delete_object(door)


            # Move living room to origin
            # for obj in living_room_objects:
            #     obj.location.x = obj.location.x - center[0]
            #     obj.location.y = obj.location.y - center[1]
            #     obj.location.z = obj.location.z - center[2]
                
            # # Adjust remaining doors' positions
            # for door in door_collection.objects:
            #     door.location.x = door.location.x - center[0]
            #     door.location.y = door.location.y - center[1]
            #     door.location.z = door.location.z - center[2]
               
   
    # p.run_stage(
    #     "room_pillars",
    #     room_dec.room_pillars,
    #     rooms_split["wall"].objects,
    #     constants,
    # )
    #print("WALLS!!!", rooms_split["wall"].names)
    p.run_stage(
        "room_walls",
        room_dec.room_walls,
        rooms_split["wall"].objects,
        constants,
        use_chance=False,
    )
    p.run_stage(
        "room_floors",
        room_dec.room_floors,
        rooms_split["floor"].objects,
        use_chance=False,
    )
    # p.run_stage(
    #     "room_ceilings",
    #     room_dec.room_ceilings,
    #     rooms_split["ceiling"].objects,
    #     use_chance=False,
    # )

    # state.print()
    state.to_json(output_folder / "solve_state.json")

    def turn_off_lights():
        for o in bpy.data.objects:
            if o.type == "LIGHT" and not o.data.cycles.is_portal:
                print(f"Deleting {o.name}")
                butil.delete(o)

    p.run_stage("lights_off", turn_off_lights)

    def invisible_room_ceilings():
        rooms_split["exterior"].hide_viewport = True
        rooms_split["exterior"].hide_render = True
        invisible_to_camera.apply(list(rooms_split["ceiling"].objects))
        invisible_to_camera.apply(
            [o for o in bpy.data.objects if "CeilingLight" in o.name]
        )

    #p.run_stage("invisible_room_ceilings", invisible_room_ceilings, use_chance=False)

    # p.run_stage(
    #     "overhead_cam",
    #     place_cam_overhead,
    #     cam=camera_rigs[0],
    #     bbox=solved_bbox,
    #     use_chance=False,
    # )

    p.run_stage(
        "hide_other_rooms",
        hide_other_rooms,
        state,
        rooms_split,
        keep_rooms=[r.name for r in solved_rooms],
        use_chance=False,
    )

    height = p.run_stage(
        "nature_backdrop",
        create_outdoor_backdrop,
        terrain,
        house_bbox=house_bbox,
        cameras=[rig.children[0] for rig in camera_rigs],
        p=p,
        params=overrides,
        use_chance=False,
        prereq="terrain",
        default=0,
    )
    move_living_room_to_origin(state)
    simplify_furniture_meshes(state, target_faces=3000)

    if overrides.get("topview", False):
        rooms_split["exterior"].hide_viewport = True
        rooms_split["ceiling"].hide_viewport = True
        rooms_split["exterior"].hide_render = True
        rooms_split["ceiling"].hide_render = True
        for group in ["wall", "floor"]:
            for wall in rooms_split[group].objects:
                for mat in wall.data.materials:
                    for n in mat.node_tree.nodes:
                        if n.type == "BSDF_PRINCIPLED":
                            n.inputs["Alpha"].default_value = overrides.get(
                                "alpha_walls", 1.0
                            )
        bbox = np.concatenate(
            [
                read_co(r) + np.array(r.location)[np.newaxis, :]
                for r in rooms_meshed.objects
            ]
        )
        camera = camera_rigs[0].children[0]
        camera_rigs[0].location = 0, 0, 0
        camera_rigs[0].rotation_euler = 0, 0, 0
        bpy.context.scene.camera = camera
        rot_x = np.deg2rad(overrides.get("topview_rot_x", 0))
        rot_z = np.deg2rad(overrides.get("topview_rot_z", 0))
        camera.rotation_euler = rot_x, 0, rot_z
        cam_x = (np.amax(bbox[:, 0]) + np.amin(bbox[:, 0])) / 2
        cam_y = (np.amax(bbox[:, 1]) + np.amin(bbox[:, 1])) / 2
        for cam_dist in np.exp(np.linspace(1.0, 5.0, 500)):
            camera.location = (
                cam_x + cam_dist * np.sin(rot_x) * np.sin(rot_z),
                cam_y - cam_dist * np.sin(rot_x) * np.cos(rot_z),
                cam_dist * np.cos(rot_x),
            )
            bpy.context.view_layer.update()
            inview = points_inview(bbox, camera)
            if inview.all():
                for area in bpy.context.screen.areas:
                    if area.type == "VIEW_3D":
                        area.spaces.active.region_3d.view_perspective = "CAMERA"
                        break
                break

    p.save_results(output_folder / "pipeline_coarse.csv")

    return {
        "height_offset": height,
        "whole_bbox": house_bbox,
    }
def debug_door_positions():
    # Get door collection
    door_collection = bpy.data.collections.get('unique_assets:doors')
    if not door_collection:
        print("Could not find door collection")
        return
        
    # Print hierarchy and positions
    for obj in door_collection.objects:
        print(f"\nDoor object: {obj.name}")
        print(f"World location: {obj.matrix_world.translation}")
        print(f"Local location: {obj.location}")
        if obj.parent:
            print(f"Parent: {obj.parent.name}")
            print(f"Parent world location: {obj.parent.matrix_world.translation}")
            print(f"Parent local location: {obj.parent.location}")
            
        # Check for spawn assets
        for child in obj.children:
            print(f"Child: {child.name}")
            print(f"Child world location: {child.matrix_world.translation}")
            print(f"Child local location: {child.location}")

def apply_height_cut(wall_obj, cut_height=0.2):
    """Keep only the lower part of wall mesh up to absolute height.
    
    Args:
        wall_obj: Blender mesh object
        cut_height: Absolute height in meters to keep
    """
    # First deselect everything
    bpy.ops.object.select_all(action='DESELECT')
    
    # Make sure we're in object mode
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Set the wall object as active and selected
    bpy.context.view_layer.objects.active = wall_obj
    wall_obj.select_set(True)
    
    # Now it's safe to enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Bisect the mesh at the absolute cut height
    bpy.ops.mesh.bisect(
        plane_co=(0, 0, cut_height),
        plane_no=(0, 0, 1),
        clear_inner=False,
        clear_outer=True,
        threshold=0.0001
    )
    
    # # Fill any holes in the cut surface
    # bpy.ops.mesh.select_all(action='SELECT')
    # bpy.ops.mesh.region_to_loop()
    # bpy.ops.mesh.edge_face_add()
    
    # Recalculate normals
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    wall_obj.select_set(False)

def get_room_object(room_name, object_type):
    """
    Get room object (floor or wall) with different naming patterns
    
    Args:
        room_name (str): Name of the room (e.g., "bedroom", "living")
        object_type (str): Type of object ("floor" or "wall")
    """
    patterns = [
        f'{room_name}-room_0/0.{object_type}',  # Pattern with -room
        f'{room_name}_0/0.{object_type}'        # Pattern without -room (bedroom style)
    ]
    
    for pattern in patterns:
        room_obj = bpy.data.objects.get(pattern)
        if room_obj:
            return room_obj
            
    raise ValueError(f"Could not find {object_type} object for room: {room_name}")

def move_living_room_to_origin(state):
    #debug_door_positions()
    # Get room objects
    floor_obj = get_room_object(ROOMTYPE, "floor") 
    wall_obj = get_room_object(ROOMTYPE, "wall") 
    apply_height_cut(wall_obj, cut_height=0.4)
    if not floor_obj or not wall_obj:
        print("Could not find floor or wall object")
        return
    
    # Get door collection
    door_collection = bpy.data.collections.get('unique_assets:doors')
    if not door_collection:
        print("Could not find door collection")
        return

    # Calculate the current location (using floor object as reference)
    current_location = floor_obj.location.copy()
    
    # Move room objects
    for obj in [floor_obj, wall_obj]:
        # Move the object
        obj.location = obj.location - current_location
        
        # Update mesh data
        if obj.data:
            translation = mathutils.Matrix.Translation(-current_location)
            obj.data.transform(translation)
    
    # Get door collection
    door_collection = bpy.data.collections.get('unique_assets:doors')
    if door_collection:
        # First move the parent door objects (they have absolute world coordinates)
        for spawn_asset in door_collection.objects:
            if spawn_asset.parent:
                # Move the parent door
                spawn_asset.parent.location = spawn_asset.parent.location - current_location
                
                # The spawn assets' local coordinates should stay the same 
                # relative to their parents, so we don't need to modify them
                
                # Update mesh data of both parent and spawn asset
                if spawn_asset.parent.data:
                    translation = mathutils.Matrix.Translation(-current_location)
                    spawn_asset.parent.data.transform(translation)
                if spawn_asset.data:
                    spawn_asset.data.transform(translation)

     # Function to move an object and its hierarchy
    def move_object_and_children(obj):
        # Get original world space locations of all children before moving anything
        original_world_locs = {}
        for child in obj.children_recursive:
            original_world_locs[child] = child.matrix_world.copy()

        # Move the parent object
        obj.location = obj.location - current_location
        if obj.data:
            translation = mathutils.Matrix.Translation(-current_location)
            obj.data.transform(translation)
            
        # Update the scene to apply parent's transformation
        bpy.context.view_layer.update()
        
        # For each child, maintain its original world position
        for child in obj.children_recursive:
            # Calculate and set the new local position that will maintain world position
            new_world_matrix = original_world_locs[child]
            new_world_matrix.translation -= current_location
            
            # Convert world matrix to local (parent space) matrix
            if child.parent:
                new_local_matrix = child.parent.matrix_world.inverted() @ new_world_matrix
                child.matrix_local = new_local_matrix
                
            # Update mesh data if it exists
            if child.data:
                translation = mathutils.Matrix.Translation(-current_location)
                child.data.transform(translation)
    # Move furniture and other objects
    spawn_to_state = {}
    for state_key, state_obj in state.objs.items():
        if hasattr(state_obj, 'obj'):  # Make sure the state object has an 'obj' attribute
            spawn_to_state[state_obj.obj.name] = state_key
    #print("SPAWN TO KEYS", spawn_to_state)
    #print(state.objs.keys())
    for obj in bpy.data.objects:
        #if obj.parent and obj.parent.name in state.objs:
        is_related = False
        if "Factory" in obj.name and obj.name in spawn_to_state:
            if "BedFactory" in obj.name and obj.type == 'MESH':
                print(f"Cleaning up mesh for {obj.name}")
                cleanup_mesh(obj)
            state_key = spawn_to_state[obj.name]
            for relation in state.objs[state_key].relations:
                print(obj.name,  relation.target_name)
                if ROOMTYPE in relation.target_name:
                    is_related = True
                    break
            
        if is_related:
            move_object_and_children(obj)
    
    # Force scene update
    bpy.context.view_layer.update()
    
    print(f"Moved living room and all doors from {current_location} to origin")

def _export_obj(filepath, selected_only=False, objects=None):
    """Helper function to export scene with consistent settings."""
    if objects:
        # Deselect all objects first
        bpy.ops.object.select_all(action='DESELECT')
        # Select only the specified objects
        for obj in objects:
            obj.select_set(True)
        selected_only = True

    bpy.ops.wm.obj_export(
        filepath=str(filepath),
        export_selected_objects=selected_only,
        forward_axis='X',
        up_axis='Z',
        export_materials=True,
        export_normals=True,
        export_uv=True,
        export_colors=True,
        export_triangulated_mesh=False,
        export_curves_as_nurbs=True,
        export_object_groups=True,
        export_material_groups=True,
        export_vertex_groups=True,
        global_scale=1.0,
        path_mode='AUTO',
        export_eval_mode='DAG_EVAL_VIEWPORT'
    )

    if objects:
        # Deselect the objects after export
        for obj in objects:
            obj.select_set(False)

# Bed frame mesh sometimes occur large flying vertices
def cleanup_mesh(obj, distance_threshold=3.0):
    """Remove vertices and their faces that are too far from object center."""
    # Enter edit mode
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Get mesh data
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    
    # Select vertices beyond threshold
    for v in bm.verts:
        if v.co.length > distance_threshold:
            v.select = True
    
    # Select linked faces
    bpy.ops.mesh.select_linked(delimit=set())
    
    # Delete selection
    bpy.ops.mesh.delete(type='VERT')
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)

def export_scene_to_obj(output_folder, room_name=ROOMTYPE):
    """
    Export the scene to OBJ format with +X forward and Z up.
    Creates three OBJ files: complete scene, scene without floor, and floor only.
    
    Args:
        output_folder (Path): Path to output directory
        room_name (str): Name of the room (e.g., "living", "dining")
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    
    # Get the floor object
    floor_obj = get_room_object(room_name, "floor") 
    if not floor_obj:
        raise ValueError(f"Could not find floor object for room: {room_name}")
    
    # Store floor visibility state
    was_visible = floor_obj.hide_viewport
    
    # Hide the floor for no-floor export
    floor_obj.hide_viewport = True
    
    # Export scene without floor
    no_floor_path = output_folder / "scene_no_floor.obj"
    _export_obj(no_floor_path)
    
    # Restore floor visibility and export just the floor
    floor_obj.hide_viewport = was_visible
    floor_path = output_folder / "floor.obj"
    _export_obj(floor_path, objects=[floor_obj])
    
    print(f"Exported scene to:")
    print(f"  - {no_floor_path}")
    print(f"  - {floor_path}")
    
    return no_floor_path, floor_path


def simplify_furniture_meshes(state, target_faces=1000):
    """Simplify chair and table meshes to reduce face count.
    
    Args:
        state: The scene state containing object relationships
        target_faces: Target number of faces for simplified meshes
    """
    # Get spawn to state mapping
    spawn_to_state = {}
    for state_key, state_obj in state.objs.items():
        if hasattr(state_obj, 'obj'):
            spawn_to_state[state_obj.obj.name] = state_key
            
    # Find and simplify furniture
    for obj in bpy.data.objects:
        if "Factory" in obj.name and obj.name in spawn_to_state:
            state_key = spawn_to_state[obj.name]
            
            # Check if object is related to dining room
            is_furniture = False
            for relation in state.objs[state_key].relations:
                if ROOMTYPE in relation.target_name:
                    # Check if this is a chair or table
                    if ("Chair" in obj.name):
                        is_furniture = True
                        break
                        
            if is_furniture:
                print(f"Simplifying mesh for {obj.name}")
                simplify_mesh(obj, target_faces)

def simplify_mesh(obj, target_faces):
    """Apply mesh decimation to reduce face count.
    
    Args:
        obj: Blender object to simplify
        target_faces: Target number of faces
    """
    # Make sure we're in object mode
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Select only this object
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Get current face count
    current_faces = len(obj.data.polygons)
    if current_faces <= target_faces:
        print(f"{obj.name} already has fewer faces ({current_faces}) than target ({target_faces})")
        return
        
    # Calculate ratio needed to reach target faces
    ratio = target_faces / current_faces
    
    # Add decimate modifier
    decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
    decimate.ratio = ratio
    decimate.use_collapse_triangulate = True
    
    # Apply the modifier
    bpy.ops.object.modifier_apply(modifier=decimate.name)
    
    # Deselect object
    obj.select_set(False)
    print(f"Simplified {obj.name} from {current_faces} to {len(obj.data.polygons)} faces")

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
    export_scene_to_obj(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--input_folder", type=Path, default=None)
    parser.add_argument(
        "-s", "--seed", default=None, help="The seed used to generate the scene"
    )
    parser.add_argument(
        "-t",
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
