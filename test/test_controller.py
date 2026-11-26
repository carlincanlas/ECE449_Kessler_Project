# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple
import math


class TestController(KesslerController):
    def __init__(self):
        self.printed_state = False

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:

        if not self.printed_state:

            print("\nSHIP POSITION:")
            print(ship_state["position"])

            print("\nSHIP HEADING:")
            print(ship_state["heading"])

            print("\nFIRST ASTEROID:")
            print(game_state.dict["asteroids"][0])

            ship_x, ship_y = ship_state["position"]

            asteroid = game_state.dict["asteroids"][0]
            asteroid_x, asteroid_y = asteroid["position"]

            distance = (
                (asteroid_x - ship_x)**2 +
                (asteroid_y - ship_y)**2
            )**0.5

            print("\nDISTANCE TO FIRST ASTEROID:")
            print(distance)

            dx = asteroid_x - ship_x
            dy = asteroid_y - ship_y

            angle_to_asteroid = math.degrees(math.atan2(dy, dx))

            print("\nANGLE TO FIRST ASTEROID:")
            print(angle_to_asteroid)

            heading = ship_state["heading"]

            angle_difference = angle_to_asteroid - heading

            # Normalize the angle to the range -180 to 180 degrees
            angle_difference = (angle_difference + 180) % 360 - 180

            print("\nANGLE DIFFERENCE:")
            print(angle_difference)

            self.printed_state = True

        thrust = 50
        turn_rate = -90
        fire = True
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
