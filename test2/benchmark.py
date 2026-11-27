# Fast, no-graphics comparison of controllers on the same games.
#
#   python benchmark.py
#
# Runs every variant below on the same 5 seeds (10 asteroids, 60 s, 3 lives, unlimited
# bullets) in parallel and prints one summary line per variant. Send me that output.

import os
import random
import time
from multiprocessing import Pool

import numpy as np

from kesslergame import Scenario, KesslerGame, GraphicsType
try:
    from kesslergame import TrainerEnvironment
except ImportError:
    TrainerEnvironment = None

import snoopy_flying_ace as sfa
from snoopy_original import SnoopyOriginal

SEEDS = [1, 2, 3, 4, 5]
NUM_ASTEROIDS = 10
TIME_LIMIT = 60

THRUST_GENES = ('rev_high', 'rev_low', 'fwd_low')


def genes_with(**overrides):
    return [overrides.get(name, default) for (name, default, _lo, _hi) in sfa.GENE_SPEC]


# name -> (kind, genes, mines_enabled)
VARIANTS = {
    'A original':               ('original', None, False),
    'B new, no thrust, no mines': ('new', genes_with(rev_high=0.0, rev_low=0.0, fwd_low=0.0), False),
    'C new, thrust, no mines':  ('new', None, False),
    'D new, thrust + mines':    ('new', None, True),
}


def make_game():
    settings = {'perf_tracker': False,
                'graphics_type': GraphicsType.NoGraphics,
                'realtime_multiplier': 0,
                'graphics_obj': None,
                'frequency': 30}
    if TrainerEnvironment is not None:
        return TrainerEnvironment(settings=settings)
    return KesslerGame(settings=settings)


def run_job(job):
    variant, seed = job
    kind, genes, mines = VARIANTS[variant]
    sfa.ENABLE_MINES = mines
    controller = SnoopyOriginal() if kind == 'original' else sfa.SnoopyFlyingAce(genes)
    random.seed(seed)
    np.random.seed(seed)
    scenario = Scenario(name=f'bench_{seed}',
                        num_asteroids=NUM_ASTEROIDS,
                        ship_states=[{'position': (400, 400), 'angle': 90, 'lives': 3,
                                      'team': 1, 'mines_remaining': 3}],
                        map_size=(1000, 800),
                        time_limit=TIME_LIMIT,
                        ammo_limit_multiplier=0,
                        stop_if_no_ammo=False)
    try:
        score, _ = make_game().run(scenario=scenario, controllers=[controller])
        team = score.teams[0]
        return (variant, seed, team.asteroids_hit, team.deaths, team.accuracy,
                getattr(score, 'sim_time', float('nan')))
    except Exception as exc:
        return (variant, seed, -1, -1, 0.0, repr(exc))


def main():
    jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        results = pool.map(run_job, jobs)

    print(f'\n{len(jobs)} games in {time.time() - t0:.0f}s\n')
    print(f'{"variant":30s} {"hits":>6s} {"deaths":>7s} {"acc":>6s} {"avg sim s":>10s}   per-seed deaths')
    for v in VARIANTS:
        rows = [r for r in results if r[0] == v]
        errors = [r[5] for r in rows if r[2] < 0]
        if errors:
            print(f'{v:30s} ERROR: {errors[0]}')
            continue
        hits = sum(r[2] for r in rows)
        deaths = sum(r[3] for r in rows)
        acc = sum(r[4] for r in rows) / len(rows)
        times = [r[5] for r in rows if isinstance(r[5], (int, float))]
        avg_t = sum(times) / len(times) if times else float('nan')
        per_seed = ' '.join(str(r[3]) for r in sorted(rows, key=lambda r: r[1]))
        print(f'{v:30s} {hits:6d} {deaths:7d} {acc:6.2f} {avg_t:10.1f}   {per_seed}')


if __name__ == '__main__':
    main()
