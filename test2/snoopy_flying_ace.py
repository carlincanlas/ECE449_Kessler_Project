# SnoopyFlyingAce - genetic fuzzy tree controller for the Kessler Game (Asteroids)
# Adapted from scott_dick_controller.py, Fall 2023, Dr. Scott Dick
#
# Three fuzzy systems, all controlling the ship every frame:
#   1. Targeting : bullet travel time + aim error     -> turn_rate, fire
#   2. Movement  : nearest-asteroid distance + speed  -> thrust
#   3. Mines     : nearest-asteroid distance + speed  -> drop_mine
#
# A genetic algorithm (train_ga.py) tunes the genes in GENE_SPEC. With the
# default genes this controller behaves exactly like the original version.

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame import KesslerController

# (name, default, low, high) -- the GA searches between low and high.
GENE_SPEC = [
    ("bullet_s",     0.05,  0.02, 0.15),   # s: bullet travel time that counts as "short"
    ("near_hi",    200.0,  100.0, 400.0),   # px: where "Near" asteroid distance ends
    ("thrust_scale", 1.0,   0.5,   5.0),    # multiplier on the fuzzy thrust output
    ("mine_near",   80.0,   40.0, 150.0),   # px: "Near" distance for mine drops
    ("mine_fast",    5.0,    2.0,  20.0),   # speed at which the ship counts as "fast" for mines
    ("mine_gap_s",   0.0,    0.0,  10.0),   # s: minimum time between mine drops
]

# Trained genes go here (same order as GENE_SPEC). Defaults = original behaviour.
# DEFAULT_GENES = [g[1] for g in GENE_SPEC]
DEFAULT_GENES = [0.0312, 137.7451, 2.8111, 89.7316, 5.0561, 0.0034]

BULLET_SPEED = 800.0
THRUST_MAX = 480.0


class SnoopyFlyingAce(KesslerController):

    def __init__(self, genes: Optional[Sequence[float]] = None):
        raw = list(genes) if genes is not None else list(DEFAULT_GENES)
        if len(raw) != len(GENE_SPEC):
            raise ValueError(f"expected {len(GENE_SPEC)} genes, got {len(raw)}")
        self.g: Dict[str, float] = {
            name: min(max(float(v), lo), hi)
            for (name, _default, lo, hi), v in zip(GENE_SPEC, raw)
        }
        self._frame = 0
        self._last_mine_frame = -10 ** 9
        self._build_targeting()
        self._build_movement()
        self._build_mines()

    def _build_targeting(self):
        s = self.g["bullet_s"]

        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi / 30, math.pi / 30, 0.002), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')

        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, s])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, s, 2 * s])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.0, 2 * s)

        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -math.pi / 30, -2 * math.pi / 90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi / 30, -2 * math.pi / 90, -math.pi / 90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2 * math.pi / 90, -math.pi / 90, math.pi / 90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-math.pi / 90, math.pi / 90, 2 * math.pi / 90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi / 90, 2 * math.pi / 90, math.pi / 30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, 2 * math.pi / 90, math.pi / 30)

        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180, -120, -60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120, -60, 60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60, 60, 120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120, 180, 180])

        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        # Same 18 rules as the original: aim at the target; fire when nearly
        # aimed, or at any angle when the bullet travel time is short.
        rules = []
        for bt in ('L', 'M', 'S'):
            for th in ('NL', 'NM', 'NS', 'PS', 'PM', 'PL'):
                fire_label = 'Y' if (bt == 'S' or th in ('NS', 'PS')) else 'N'
                rules.append(ctrl.Rule(bullet_time[bt] & theta_delta[th],
                                       (ship_turn[th], ship_fire[fire_label])))

        self.targeting_control = ctrl.ControlSystem(rules)
        self.targeting_sim = ctrl.ControlSystemSimulation(self.targeting_control)

    def _build_movement(self):
        near_hi = self.g["near_hi"]

        dist = ctrl.Antecedent(np.arange(0, 801, 1), 'asteroid_distance')
        speed = ctrl.Antecedent(np.arange(0, 401, 1), 'ship_speed')
        thrust = ctrl.Consequent(np.arange(0, 20.1, 0.1), 'ship_thrust')

        dist['Near'] = fuzz.trimf(dist.universe, [0, 0, near_hi])
        dist['Medium'] = fuzz.trimf(dist.universe, [100, 300, 500])
        dist['Far'] = fuzz.trimf(dist.universe, [350, 800, 800])

        speed['Slow'] = fuzz.trimf(speed.universe, [0, 0, 100])
        speed['Medium'] = fuzz.trimf(speed.universe, [50, 150, 250])
        speed['Fast'] = fuzz.trimf(speed.universe, [200, 400, 400])

        thrust['Zero'] = fuzz.trimf(thrust.universe, [0, 0, 5])
        thrust['Low'] = fuzz.trimf(thrust.universe, [0, 5, 10])
        thrust['Medium'] = fuzz.trimf(thrust.universe, [5, 10, 15])
        thrust['High'] = fuzz.trimf(thrust.universe, [10, 15, 20])

        rules = [
            ctrl.Rule(dist['Near'] & speed['Slow'], thrust['Zero']),
            ctrl.Rule(dist['Near'] & speed['Medium'], thrust['Zero']),
            ctrl.Rule(dist['Near'] & speed['Fast'], thrust['Zero']),
            ctrl.Rule(dist['Medium'] & speed['Slow'], thrust['Medium']),
            ctrl.Rule(dist['Medium'] & speed['Medium'], thrust['Low']),
            ctrl.Rule(dist['Medium'] & speed['Fast'], thrust['Zero']),
            ctrl.Rule(dist['Far'] & speed['Slow'], thrust['High']),
            ctrl.Rule(dist['Far'] & speed['Medium'], thrust['Medium']),
            ctrl.Rule(dist['Far'] & speed['Fast'], thrust['Low']),
        ]
        self.movement_control = ctrl.ControlSystem(rules)
        self.movement_sim = ctrl.ControlSystemSimulation(self.movement_control)

    def _build_mines(self):
        mn = self.g["mine_near"]
        mf = self.g["mine_fast"]

        mdist = ctrl.Antecedent(np.arange(0, 801, 1), 'mine_distance')
        mspeed = ctrl.Antecedent(np.arange(0, 41, 1), 'mine_speed')
        mout = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'mine_output')

        mdist['Near'] = fuzz.trimf(mdist.universe, [0, 0, mn])
        mdist['Far'] = fuzz.trimf(mdist.universe, [0.5 * mn, 800, 800])

        mspeed['Slow'] = fuzz.trimf(mspeed.universe, [0, 0, 2 * mf])
        mspeed['Fast'] = fuzz.trimf(mspeed.universe, [mf, 40, 40])

        mout['No'] = fuzz.trimf(mout.universe, [0, 0, 0.5])
        mout['Yes'] = fuzz.trimf(mout.universe, [0.5, 1, 1])

        rules = [
            ctrl.Rule(mdist['Near'] & mspeed['Slow'], mout['No']),
            ctrl.Rule(mdist['Near'] & mspeed['Fast'], mout['Yes']),
            ctrl.Rule(mdist['Far'], mout['No']),
        ]
        self.mine_control = ctrl.ControlSystem(rules)
        self.mine_sim = ctrl.ControlSystemSimulation(self.mine_control)

    @staticmethod
    def _run(sim, inputs: Dict[str, float], out: str, default: float) -> float:
        """Run one fuzzy system; fall back to `default` if no rule fires."""
        try:
            for key, value in inputs.items():
                sim.input[key] = float(value)
            sim.compute()
            return float(sim.output[out])
        except Exception:
            return default

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        self._frame += 1
        asteroids = game_state["asteroids"]
        if not asteroids:
            return 0.0, 0.0, False, False

        ship_x, ship_y = ship_state["position"][0], ship_state["position"][1]

        nearest = None
        nearest_d = float('inf')
        for a in asteroids:
            d = math.hypot(a["position"][0] - ship_x, a["position"][1] - ship_y)
            if d < nearest_d:
                nearest, nearest_d = a, d

        ship_speed = math.hypot(ship_state["velocity"][0], ship_state["velocity"][1])

        # Thrust (fuzzy) 
        thrust = self._run(
            self.movement_sim,
            {'asteroid_distance': min(nearest_d, 800.0),
             'ship_speed': min(ship_speed, 400.0)},
            'ship_thrust', 0.0)
        thrust = max(-THRUST_MAX, min(THRUST_MAX, thrust * self.g["thrust_scale"]))

        # Mines (fuzzy) 
        drop_mine = False
        try:
            mines_left = ship_state["mines_remaining"]
        except (KeyError, AttributeError, TypeError):
            mines_left = 0
        gap_frames = int(self.g["mine_gap_s"] * 30)
        if mines_left > 0 and self._frame - self._last_mine_frame >= gap_frames:
            m = self._run(
                self.mine_sim,
                {'mine_distance': min(nearest_d, 800.0),
                 'mine_speed': min(ship_speed, 40.0)},
                'mine_output', 0.0)
            if m >= 0.5:
                drop_mine = True
                self._last_mine_frame = self._frame

        # Targeting (intercept time from relative motion)
        ax, ay = nearest["position"][0], nearest["position"][1]
        avx, avy = nearest["velocity"][0], nearest["velocity"][1]
        rel_x, rel_y = ax - ship_x, ay - ship_y

        a = avx ** 2 + avy ** 2 - BULLET_SPEED ** 2
        b = 2 * (rel_x * avx + rel_y * avy)
        c = rel_x ** 2 + rel_y ** 2
        bullet_t = 0.0

        if abs(a) > 1e-9:
            disc = b ** 2 - 4 * a * c
            if disc >= 0:
                root = math.sqrt(disc)
                positive = [t for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)) if t > 0]
                if positive:
                    bullet_t = min(positive)
        elif abs(b) > 1e-9:
            t = -c / b
            if t > 0:
                bullet_t = t

        t_hit = bullet_t + 1 / 30
        target_angle = math.atan2(ay + avy * t_hit - ship_y, ax + avx * t_hit - ship_x)
        theta = target_angle - math.radians(ship_state["heading"])
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        bullet_t = float(np.clip(bullet_t, 0.0, 0.998))
        theta_in = float(np.clip(theta, -math.pi / 30, math.pi / 30 - 0.002))

        try:
            sim = self.targeting_sim
            sim.input['bullet_time'] = bullet_t
            sim.input['theta_delta'] = theta_in
            sim.compute()
            turn_rate = float(sim.output['ship_turn'])
            fire = float(sim.output['ship_fire']) >= 0
        except Exception:
            turn_rate = math.copysign(180.0, theta)
            fire = False

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

    @property
    def name(self) -> str:
        return "SnoopyFlyingAce"
