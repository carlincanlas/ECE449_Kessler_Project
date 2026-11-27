# Adapted from scott_dick_controller.py, Fall 2023, Dr. Scott Dick

from kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np


class SnoopyFlyingAce(KesslerController):

    def __init__(self):
        # TARGETING FUZZY CONTROL SYSTEM

        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi/30, math.pi/30, 0.002), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')

        # Bullet travel time: Short, Medium, Long
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.0, 0.1)

        # Angular error: Negative Large to Positive Large (radians)
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -math.pi/30, -2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi/30, -2*math.pi/90, -math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90, -math.pi/90, math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-math.pi/90, math.pi/90, 2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90, 2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, 2*math.pi/90, math.pi/30)

        # Turn rate: Degrees per second
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180, -120, -60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120, -60, 60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60, 60, 120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120, 180, 180])

        # Fire decision
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        # Targeting rules
        targeting_rules = [
            ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
        ]

        self.targeting_control = ctrl.ControlSystem(targeting_rules)

        # MOVEMENT FUZZY CONTROL SYSTEM

        asteroid_distance = ctrl.Antecedent(np.arange(0, 801, 1), 'asteroid_distance')
        ship_speed = ctrl.Antecedent(np.arange(0, 401, 1), 'ship_speed')
        ship_thrust = ctrl.Consequent(np.arange(0, 20.1, 0.1), 'ship_thrust')

        # Asteroid distance: Near, Medium, Far
        asteroid_distance['Near'] = fuzz.trimf(asteroid_distance.universe, [0, 0, 200])
        asteroid_distance['Medium'] = fuzz.trimf(asteroid_distance.universe, [100, 300, 500])
        asteroid_distance['Far'] = fuzz.trimf(asteroid_distance.universe, [350, 800, 800])

        # Ship speed: Slow, Medium, Fast
        ship_speed['Slow'] = fuzz.trimf(ship_speed.universe, [0, 0, 100])
        ship_speed['Medium'] = fuzz.trimf(ship_speed.universe, [50, 150, 250])
        ship_speed['Fast'] = fuzz.trimf(ship_speed.universe, [200, 400, 400])

        # Thrust: Zero, Low, Medium, High
        ship_thrust['Zero'] = fuzz.trimf(ship_thrust.universe, [0, 0, 5])
        ship_thrust['Low'] = fuzz.trimf(ship_thrust.universe, [0, 5, 10])
        ship_thrust['Medium'] = fuzz.trimf(ship_thrust.universe, [5, 10, 15])
        ship_thrust['High'] = fuzz.trimf(ship_thrust.universe, [10, 15, 20])

        movement_rules = [
            ctrl.Rule(asteroid_distance['Near'] & ship_speed['Slow'], ship_thrust['Zero']),
            ctrl.Rule(asteroid_distance['Near'] & ship_speed['Medium'], ship_thrust['Zero']),
            ctrl.Rule(asteroid_distance['Near'] & ship_speed['Fast'], ship_thrust['Zero']),
            ctrl.Rule(asteroid_distance['Medium'] & ship_speed['Slow'], ship_thrust['Medium']),
            ctrl.Rule(asteroid_distance['Medium'] & ship_speed['Medium'], ship_thrust['Low']),
            ctrl.Rule(asteroid_distance['Medium'] & ship_speed['Fast'], ship_thrust['Zero']),
            ctrl.Rule(asteroid_distance['Far'] & ship_speed['Slow'], ship_thrust['High']),
            ctrl.Rule(asteroid_distance['Far'] & ship_speed['Medium'], ship_thrust['Medium']),
            ctrl.Rule(asteroid_distance['Far'] & ship_speed['Fast'], ship_thrust['Low'])
        ]

        self.movement_control = ctrl.ControlSystem(movement_rules)

        # MINE FUZZY CONTROL SYSTEM

        mine_distance = ctrl.Antecedent(np.arange(0, 801, 1), 'mine_distance')
        mine_speed = ctrl.Antecedent(np.arange(0, 41, 1), 'mine_speed')
        mine_output = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'mine_output')

        # Distance: Near, Far
        mine_distance['Near'] = fuzz.trimf(mine_distance.universe, [0, 0, 80])
        mine_distance['Far'] = fuzz.trimf(mine_distance.universe, [40, 800, 800])

        # Speed: Slow, Fast
        mine_speed['Slow'] = fuzz.trimf(mine_speed.universe, [0, 0, 10])
        mine_speed['Fast'] = fuzz.trimf(mine_speed.universe, [5, 40, 40])

        # Mine decision
        mine_output['No'] = fuzz.trimf(mine_output.universe, [0, 0, 0.5])
        mine_output['Yes'] = fuzz.trimf(mine_output.universe, [0.5, 1, 1])

        mine_rules = [
            ctrl.Rule(mine_distance['Near'] & mine_speed['Slow'], mine_output['No']),
            ctrl.Rule(mine_distance['Near'] & mine_speed['Fast'], mine_output['Yes']),
            ctrl.Rule(mine_distance['Far'], mine_output['No'])
        ]

        self.mine_control = ctrl.ControlSystem(mine_rules)

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # Find the closest asteroid
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None

        for asteroid in game_state["asteroids"]:
            curr_dist = math.hypot(
                ship_pos_x - asteroid["position"][0],
                ship_pos_y - asteroid["position"][1]
            )
            if closest_asteroid is None or curr_dist < closest_asteroid["dist"]:
                closest_asteroid = {"aster": asteroid, "dist": curr_dist}

        # Calculate ship speed and nearest asteroid distance
        ship_speed_value = math.hypot(
            ship_state["velocity"][0],
            ship_state["velocity"][1]
        )
        asteroid_distance_value = closest_asteroid["dist"] if closest_asteroid is not None else 800.0

        # Evaluate movement fuzzy system
        movement = ctrl.ControlSystemSimulation(self.movement_control, flush_after_run=1)
        movement.input['asteroid_distance'] = np.clip(asteroid_distance_value, 0, 800)
        movement.input['ship_speed'] = np.clip(ship_speed_value, 0, 400)
        movement.compute()
        thrust = movement.output['ship_thrust']

        # Evaluate mine fuzzy system
        drop_mine = False
        if closest_asteroid is not None and ship_state["mines_remaining"] > 0:
            mine = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
            mine.input['mine_distance'] = np.clip(closest_asteroid["dist"], 0, 800)
            mine.input['mine_speed'] = np.clip(ship_speed_value, 0, 40)
            mine.compute()
            drop_mine = bool(mine.output['mine_output'] >= 0.5)

        # No target available
        if closest_asteroid is None:
            return thrust, 0.0, False, False

        # Calculate bullet intercept time using relative position and velocity
        asteroid_x = closest_asteroid["aster"]["position"][0]
        asteroid_y = closest_asteroid["aster"]["position"][1]
        asteroid_vx = closest_asteroid["aster"]["velocity"][0]
        asteroid_vy = closest_asteroid["aster"]["velocity"][1]

        relative_x = asteroid_x - ship_pos_x
        relative_y = asteroid_y - ship_pos_y
        bullet_speed = 800.0

        a = asteroid_vx**2 + asteroid_vy**2 - bullet_speed**2
        b = 2 * (relative_x * asteroid_vx + relative_y * asteroid_vy)
        c = relative_x**2 + relative_y**2
        bullet_t = 0.0

        if abs(a) > 1e-9:
            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                root = math.sqrt(discriminant)
                roots = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
                positive_roots = [t for t in roots if t > 0]
                if positive_roots:
                    bullet_t = min(positive_roots)
        elif abs(b) > 1e-9:
            possible_time = -c / b
            if possible_time > 0:
                bullet_t = possible_time

        # Predict asteroid position at bullet arrival
        intercept_time = bullet_t + 1/30
        intercept_x = asteroid_x + asteroid_vx * intercept_time
        intercept_y = asteroid_y + asteroid_vy * intercept_time

        target_angle = math.atan2(intercept_y - ship_pos_y, intercept_x - ship_pos_x)
        shooting_theta = target_angle - math.radians(ship_state["heading"])
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Keep targeting inputs within fuzzy universes
        bullet_t = np.clip(bullet_t, 0.0, 0.998)
        shooting_theta = np.clip(shooting_theta, -math.pi/30, math.pi/30 - 0.002)

        # Evaluate targeting fuzzy system
        shooting = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.compute()

        turn_rate = shooting.output['ship_turn']
        fire = bool(shooting.output['ship_fire'] >= 0)

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "SnoopyFlyingAce"
