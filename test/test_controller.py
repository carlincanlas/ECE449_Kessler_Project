# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple
import math


class TestController(KesslerController):
    def __init__(self):
        self.debug_counter = 0

    def actions(self, ship_state: Dict, game_state: Dict):

        self.debug_counter += 1

        if self.debug_counter % 30 == 0:
            print("Current heading:", ship_state["heading"])

        ship_x, ship_y = ship_state["position"]
        heading = ship_state["heading"]

        asteroid = game_state.dict["asteroids"][0]
        asteroid_x, asteroid_y = asteroid["position"]

        dx = asteroid_x - ship_x
        dy = asteroid_y - ship_y

        distance = (dx**2 + dy**2)**0.5

        angle_to_asteroid = math.degrees(math.atan2(dy, dx))

        angle_difference = (angle_to_asteroid - heading + 180) % 360 - 180

        thrust = 0
        turn_rate = 60
        fire = False
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Test Controller"

    # @property
    # def custom_sprite_path(self) -> str:
    #     return "Neo.png"
