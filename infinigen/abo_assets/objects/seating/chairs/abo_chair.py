# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import bpy
import json
from pathlib import Path
from numpy.random import choice, uniform


from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.seating.chairs.seats.curvy_seats import (
    generate_curvy_seats,
)

from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory


class ABOChairFactory(AssetFactory):
    
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.width = uniform(0.4, 0.5)
            self.size = uniform(0.38, 0.45)
            self.thickness = uniform(0.04, 0.08)
            self.bevel_width = self.thickness * (0.1 if uniform() < 0.4 else 0.5)
            self.seat_back = uniform(0.7, 1.0) if uniform() < 0.75 else 1.0
            self.seat_mid = uniform(0.7, 0.8)
            self.seat_mid_x = uniform(
                self.seat_back + self.seat_mid * (1 - self.seat_back), 1
            )
            self.seat_mid_z = uniform(0, 0.5)
            self.seat_front = uniform(1.0, 1.2)
            self.is_seat_round = uniform() < 0.6
            self.is_seat_subsurf = uniform() < 0.5

            self.asset_path = Path("/home/chenlu/ezone/infinigen/data/meshes/chair") / self.factory_seed

    def finalize_assets(self, assets):
        pass

    def create_asset(self, obj, axis, thickness=None):
        
        return obj