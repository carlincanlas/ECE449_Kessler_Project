# SnoopyFlyingAce - fuzzy controller for the Kessler Game (Asteroids)
# Adapted from scott_dick_controller.py (Dr. Scott Dick, Fall 2023)
#
# Three fuzzy systems, all tuned by one list of numbers ("genes") that the
# genetic algorithm in train_ga.py optimizes:
#   1. Targeting : bullet travel time + aim error  -> turn_rate, fire
#   2. Movement  : nearest-asteroid distance + forward velocity -> thrust
#   3. Mines     : nearest-asteroid distance + crowd size       -> drop_mine
#
# After training, paste the printed list into DEFAULT_GENES below so this
# file works on its own (the grader will construct SnoopyFlyingAce() with no
# arguments).

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame import KesslerController

# (name, default, low, high) -- the GA searches between low and high.
GENE_SPEC = [
    ("near_hi",       200.0, 100.0, 400.0),   # px: where "Near" asteroid distance ends
    ("mid_c",         350.0, 250.0, 500.0),   # px: centre of "Medium" distance
    ("vz",             60.0,  20.0, 120.0),   # m/s: width of "zero" forward speed
    ("rev_high",      150.0,   0.0, 480.0),   # strong reverse thrust magnitude (0 = never move)
    ("rev_low",         60.0,   0.0, 300.0),   # gentle reverse thrust magnitude
    ("fwd_low",         40.0,   0.0, 300.0),   # gentle forward thrust magnitude
    ("mine_near",     150.0,  60.0, 300.0),   # px: "Near" distance for mine drops
    ("crowd_radius",  250.0, 150.0, 400.0),   # px: radius used to count nearby asteroids
    ("crowd_many",      3.0,   1.0,   6.0),   # asteroid count that counts as "many"
    ("mine_thresh",     0.5,   0.3,   0.7),   # defuzzified value needed to drop a mine
    ("mine_gap_s",      5.0,   3.0,  10.0),   # seconds between mine drops (must exceed the 2 s escape)
    ("fire_cone_deg",  15.0,   3.0, 180.0),   # only fire if aim error is under this (180 = no gate)
    ("bullet_s",       0.05,  0.02,  0.15),   # s: bullet travel time that counts as "short"
]

# Paste the GA result here after training (same order as GENE_SPEC).
DEFAULT_GENES = [181.762, 305.6828, 46.9648, 98.5501, 35.3393, 50.6274, 168.5273, 210.2783, 2.6307, 0.5188, 5.0481,8.1284, 0.02]

# Set to False to turn mines off completely (handy for finding what kills the ship).
ENABLE_MINES = True

# Mine escape manoeuvre (heading locked, no shooting while it runs):
#   phase 1: reverse at ESCAPE_THRUST until ESCAPE_DIST px from the mine
#   phase 2: brake to a stop using the measured speed (never accelerates forward)
# Afterwards, while the mine is live, the ship never speeds up near it (see actions()).
ESCAPE_THRUST = 240.0
ESCAPE_DIST = 220.0         # px away from the mine before braking starts
ESCAPE_MAX_FRAMES = 45      # safety limit for phase 1
BRAKE_MAX_FRAMES = 45       # safety limit for phase 2
MINE_LIVE_FRAMES = 100      # ~3 s fuse plus the explosion, at 30 Hz
MINE_KEEP_OUT = 300.0       # px: no forward acceleration within this range of a live mine
MINE_MAX_DROP_SPEED = 60.0  # only drop a mine when the ship is nearly stationary, so the escape works

BULLET_SPEED = 800.0
THRUST_MAX = 480.0
SPEED_MAX = 240.0
MAX_DIST = 800.0
MINE_MAX_DIST = 600.0
CROWD_MAX = 10


def _tri(peak: float, half: float, lo: float, hi: float):
    """Triangle centred on `peak`, clipped to the universe [lo, hi]."""
    return [max(peak - half, lo), peak, min(peak + half, hi)]


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
        self.escape_phase = 0       # 0 = none, 1 = backing away, 2 = braking
        self.escape_frames = 0      # frames spent in the current phase
        self.mine_pos = (0.0, 0.0)  # where the last mine was dropped
        self.mine_drop_frame = -10 ** 9
        self._build_targeting()
        self._build_movement()
        self._build_mines()

    # ------------------------------------------------------------------
    # 1. Targeting (turn_rate + fire), same structure as Dr. Dick's system
    # ------------------------------------------------------------------
    def _build_targeting(self):
        s = self.g["bullet_s"]

        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi / 30, math.pi / 30, 0.002), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')

        # Bullet travel time: Short / Medium / Long
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, s])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, s, 2 * s])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.0, 2 * s)

        # Aim error in radians (one game step turns at most pi/30)
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -math.pi / 30, -2 * math.pi / 90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-math.pi / 30, -2 * math.pi / 90, -math.pi / 90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2 * math.pi / 90, -math.pi / 90, math.pi / 90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-math.pi / 90, math.pi / 90, 2 * math.pi / 90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi / 90, 2 * math.pi / 90, math.pi / 30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, 2 * math.pi / 90, math.pi / 30)

        # Turn rate (deg/s)
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180, -120, -60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120, -60, 60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60, 60, 120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120, 180, 180])

        # Fire: -1 = no, +1 = yes (thresholded at 0)
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        rules = []
        for bt in ('L', 'M', 'S'):
            for th in ('NL', 'NM', 'NS', 'PS', 'PM', 'PL'):
                # Short shots fire at any angle; otherwise fire only when nearly aimed.
                fire_label = 'Y' if (bt == 'S' or th in ('NS', 'PS')) else 'N'
                rules.append(ctrl.Rule(bullet_time[bt] & theta_delta[th],
                                       (ship_turn[th], ship_fire[fire_label])))

        self.targeting_control = ctrl.ControlSystem(rules)
        self.targeting_sim = ctrl.ControlSystemSimulation(self.targeting_control)

    # ------------------------------------------------------------------
    # 2. Movement (thrust). Thrust acts along the ship's heading, and the
    #    heading follows the nearest asteroid, so negative thrust = back away.
    # ------------------------------------------------------------------
    def _build_movement(self):
        g = self.g
        near_hi, mid_c, vz = g["near_hi"], g["mid_c"], g["vz"]

        dist = ctrl.Antecedent(np.arange(0, MAX_DIST + 1, 1), 'ast_dist')
        vfwd = ctrl.Antecedent(np.arange(-SPEED_MAX, SPEED_MAX + 1, 1), 'v_fwd')
        thrust = ctrl.Consequent(np.arange(-THRUST_MAX, THRUST_MAX + 1, 1), 'ship_thrust')

        dist['Near'] = fuzz.trimf(dist.universe, [0, 0, near_hi])
        dist['Medium'] = fuzz.trimf(dist.universe,
                                    [0.5 * near_hi, mid_c, min(2 * mid_c - 0.5 * near_hi, MAX_DIST)])
        dist['Far'] = fuzz.trimf(dist.universe, [mid_c, MAX_DIST, MAX_DIST])

        # Velocity along the heading: Neg = drifting backwards, Pos = drifting forwards
        vfwd['Neg'] = fuzz.zmf(vfwd.universe, -vz, 0)
        vfwd['Zero'] = fuzz.trimf(vfwd.universe, [-vz, 0, vz])
        vfwd['Pos'] = fuzz.smf(vfwd.universe, 0, vz)

        thrust['RevHigh'] = fuzz.trimf(thrust.universe, _tri(-g["rev_high"], 80, -THRUST_MAX, THRUST_MAX))
        thrust['RevLow'] = fuzz.trimf(thrust.universe, _tri(-g["rev_low"], 80, -THRUST_MAX, THRUST_MAX))
        thrust['Zero'] = fuzz.trimf(thrust.universe, _tri(0.0, 60, -THRUST_MAX, THRUST_MAX))
        thrust['FwdLow'] = fuzz.trimf(thrust.universe, _tri(g["fwd_low"], 80, -THRUST_MAX, THRUST_MAX))

        rules = [
            # Close to an asteroid: get away from it (unless already drifting away)
            ctrl.Rule(dist['Near'] & vfwd['Neg'], thrust['Zero']),
            ctrl.Rule(dist['Near'] & vfwd['Zero'], thrust['RevHigh']),
            ctrl.Rule(dist['Near'] & vfwd['Pos'], thrust['RevHigh']),
            # Middle distance: damp the motion
            ctrl.Rule(dist['Medium'] & vfwd['Neg'], thrust['FwdLow']),
            ctrl.Rule(dist['Medium'] & vfwd['Zero'], thrust['Zero']),
            ctrl.Rule(dist['Medium'] & vfwd['Pos'], thrust['RevLow']),
            # Far away: sit still, brake any backwards drift
            ctrl.Rule(dist['Far'] & vfwd['Neg'], thrust['FwdLow']),
            ctrl.Rule(dist['Far'] & vfwd['Zero'], thrust['Zero']),
            ctrl.Rule(dist['Far'] & vfwd['Pos'], thrust['Zero']),
        ]
        self.movement_control = ctrl.ControlSystem(rules)
        self.movement_sim = ctrl.ControlSystemSimulation(self.movement_control)

    # ------------------------------------------------------------------
    # 3. Mines: drop one when asteroids are crowding the ship
    # ------------------------------------------------------------------
    def _build_mines(self):
        g = self.g
        mn = g["mine_near"]
        many = g["crowd_many"]

        mdist = ctrl.Antecedent(np.arange(0, MINE_MAX_DIST + 1, 1), 'mine_dist')
        crowd = ctrl.Antecedent(np.arange(0, CROWD_MAX + 1, 1), 'crowd')
        mout = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'mine_out')

        mdist['Near'] = fuzz.trimf(mdist.universe, [0, 0, mn])
        mdist['Mid'] = fuzz.trimf(mdist.universe, [0.5 * mn, 1.5 * mn, min(3 * mn, MINE_MAX_DIST)])
        mdist['Far'] = fuzz.trimf(mdist.universe, [1.5 * mn, MINE_MAX_DIST, MINE_MAX_DIST])

        crowd['Few'] = fuzz.zmf(crowd.universe, 0.5 * many, 1.5 * many)
        crowd['Many'] = fuzz.smf(crowd.universe, 0.5 * many, 1.5 * many)

        mout['No'] = fuzz.trimf(mout.universe, [0, 0, 0.5])
        mout['Yes'] = fuzz.trimf(mout.universe, [0.5, 1, 1])

        # Safety comes from the escape manoeuvre in actions(), not from these rules.
        rules = [
            ctrl.Rule(mdist['Near'] & crowd['Many'], mout['Yes']),
            ctrl.Rule(mdist['Mid'] & crowd['Many'], mout['Yes']),
            ctrl.Rule(crowd['Few'], mout['No']),
            ctrl.Rule(mdist['Far'], mout['No']),
        ]
        self.mine_control = ctrl.ControlSystem(rules)
        self.mine_sim = ctrl.ControlSystemSimulation(self.mine_control)

    # ------------------------------------------------------------------
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

        # Nearest asteroid, and how many asteroids are crowding the ship
        nearest = None
        nearest_d = float('inf')
        crowd_count = 0
        for a in asteroids:
            d = math.hypot(a["position"][0] - ship_x, a["position"][1] - ship_y)
            if d < nearest_d:
                nearest, nearest_d = a, d
            if d <= self.g["crowd_radius"]:
                crowd_count += 1

        # ---- Thrust -------------------------------------------------
        heading = math.radians(ship_state["heading"])
        vx, vy = ship_state["velocity"][0], ship_state["velocity"][1]
        v_fwd = vx * math.cos(heading) + vy * math.sin(heading)
        thrust = self._run(
            self.movement_sim,
            {'ast_dist': min(nearest_d, MAX_DIST),
             'v_fwd': max(-SPEED_MAX, min(SPEED_MAX, v_fwd))},
            'ship_thrust', 0.0)
        thrust = max(-THRUST_MAX, min(THRUST_MAX, thrust))

        # ---- Mines --------------------------------------------------
        drop_mine = False
        try:
            mines_left = ship_state["mines_remaining"]
        except (KeyError, AttributeError, TypeError):
            mines_left = 0
        gap_frames = int(self.g["mine_gap_s"] * 30)
        if (ENABLE_MINES and mines_left > 0 and self.escape_phase == 0
                and math.hypot(vx, vy) <= MINE_MAX_DROP_SPEED
                and self._frame - self._last_mine_frame >= gap_frames):
            m = self._run(
                self.mine_sim,
                {'mine_dist': min(nearest_d, MINE_MAX_DIST),
                 'crowd': min(crowd_count, CROWD_MAX)},
                'mine_out', 0.0)
            if m >= self.g["mine_thresh"]:
                drop_mine = True
                self._last_mine_frame = self._frame
                self.escape_phase = 1                     # start the escape manoeuvre now
                self.escape_frames = 0
                self.mine_pos = (ship_x, ship_y)
                self.mine_drop_frame = self._frame

        # ---- Targeting (intercept time from relative motion) --------
        ax, ay = nearest["position"][0], nearest["position"][1]
        avx, avy = nearest["velocity"][0], nearest["velocity"][1]
        rel_x, rel_y = ax - ship_x, ay - ship_y

        a = avx ** 2 + avy ** 2 - BULLET_SPEED ** 2
        b = 2 * (rel_x * avx + rel_y * avy)
        c = rel_x ** 2 + rel_y ** 2

        bullet_t = nearest_d / BULLET_SPEED     # fallback: aim straight at it
        if abs(a) > 1e-9:
            disc = b ** 2 - 4 * a * c
            if disc >= 0:
                root = math.sqrt(disc)
                positive = [t for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)) if t > 0]
                if positive:
                    bullet_t = min(positive)

        t_hit = bullet_t + 1 / 30
        target_angle = math.atan2(ay + avy * t_hit - ship_y, ax + avx * t_hit - ship_x)
        theta = target_angle - heading
        theta = (theta + math.pi) % (2 * math.pi) - math.pi     # wrap to (-pi, pi)

        bullet_t = float(np.clip(bullet_t, 0.0, 0.998))
        theta_in = float(np.clip(theta, -math.pi / 30, math.pi / 30 - 0.002))

        try:
            sim = self.targeting_sim
            sim.input['bullet_time'] = bullet_t
            sim.input['theta_delta'] = theta_in
            sim.compute()
            turn_rate = float(sim.output['ship_turn'])
            fuzzy_fire = float(sim.output['ship_fire']) >= 0
        except Exception:
            turn_rate = math.copysign(180.0, theta)
            fuzzy_fire = False

        # Extra safety gate: don't waste shots when clearly off target
        fire = fuzzy_fire and abs(theta) <= math.radians(self.g["fire_cone_deg"])

        brake = max(-ESCAPE_THRUST, min(ESCAPE_THRUST, -v_fwd * 30.0))   # cancels speed along the heading

        if self.escape_phase == 1:
            # Back away from our own mine, heading locked.
            self.escape_frames += 1
            thrust = -ESCAPE_THRUST
            turn_rate = 0.0
            fire = False
            away = math.hypot(ship_x - self.mine_pos[0], ship_y - self.mine_pos[1])
            if away >= ESCAPE_DIST or self.escape_frames >= ESCAPE_MAX_FRAMES:
                self.escape_phase = 2
                self.escape_frames = 0
        elif self.escape_phase == 2:
            # Brake to a standstill (react to the real speed; drag makes timing guesses wrong).
            self.escape_frames += 1
            thrust = brake
            turn_rate = 0.0
            fire = False
            if abs(v_fwd) < 5.0 or self.escape_frames >= BRAKE_MAX_FRAMES:
                self.escape_phase = 0
        elif self._frame - self.mine_drop_frame < MINE_LIVE_FRAMES:
            # Mine still live: don't let the normal dodging move us back toward it.
            if math.hypot(ship_x - self.mine_pos[0], ship_y - self.mine_pos[1]) < MINE_KEEP_OUT:
                thrust = brake

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

    @property
    def name(self) -> str:
        return "SnoopyFlyingAce"
