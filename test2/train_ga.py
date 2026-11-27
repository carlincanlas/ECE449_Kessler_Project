# Genetic algorithm that tunes the fuzzy controller in snoopy_flying_ace.py
#
# Usage:   python train_ga.py            (defaults: 12 individuals, 6 generations)
#          python train_ga.py 16 10      (population size, generations)
#
# Best genes so far are saved to best_genes.json after EVERY generation, so you
# can press Ctrl+C at any time and still use the result.

import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

from kesslergame import Scenario, KesslerGame, GraphicsType
try:
    from kesslergame import TrainerEnvironment      # fast, no graphics
except ImportError:
    TrainerEnvironment = None

from snoopy_flying_ace import SnoopyFlyingAce, GENE_SPEC, DEFAULT_GENES

# (number of asteroids, random seed). Same cases for every individual so the
# comparison between individuals is fair.
TRAIN_CASES = [(8, 1), (12, 2)]
TRAIN_TIME_LIMIT = 60          # simulated seconds per game; matches the real tests, so deaths show up


def make_game():
    settings = {'perf_tracker': False,
                'graphics_type': GraphicsType.NoGraphics,
                'realtime_multiplier': 0,
                'graphics_obj': None,
                'frequency': 30}
    if TrainerEnvironment is not None:
        return TrainerEnvironment(settings=settings)
    return KesslerGame(settings=settings)


def make_scenario(num_asteroids, time_limit):
    return Scenario(name=f'train_{num_asteroids}',
                    num_asteroids=num_asteroids,
                    ship_states=[
                        {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3},
                    ],
                    map_size=(1000, 800),
                    time_limit=time_limit,
                    ammo_limit_multiplier=0,      # unlimited bullets
                    stop_if_no_ammo=False)


def evaluate(genes):
    """Average fitness of one gene list over the training cases."""
    total = 0.0
    try:
        for num_asteroids, seed in TRAIN_CASES:
            random.seed(seed)
            np.random.seed(seed)
            game = make_game()
            score, _ = game.run(scenario=make_scenario(num_asteroids, TRAIN_TIME_LIMIT),
                                controllers=[SnoopyFlyingAce(genes)])
            team = score.teams[0]
            # Reward kills and accuracy, punish deaths.
            total += team.asteroids_hit + 20.0 * team.accuracy - 15.0 * team.deaths
        return total / len(TRAIN_CASES)
    except Exception as exc:
        print('evaluation failed:', repr(exc))
        return -1000.0


# ---------------------------------------------------------------- GA ops
def clip_genes(genes):
    return [min(max(v, lo), hi) for v, (_n, _d, lo, hi) in zip(genes, GENE_SPEC)]


def random_near_default(sigma=0.15):
    return clip_genes([d + random.gauss(0, sigma * (hi - lo)) for (_n, d, lo, hi) in GENE_SPEC])


def crossover(a, b, alpha=0.25):
    child = []
    for x, y in zip(a, b):
        lo, hi = min(x, y), max(x, y)
        span = hi - lo
        child.append(random.uniform(lo - alpha * span, hi + alpha * span))
    return clip_genes(child)


def mutate(genes, rate=0.25, sigma=0.12):
    out = list(genes)
    for i, (_n, _d, lo, hi) in enumerate(GENE_SPEC):
        if random.random() < rate:
            out[i] += random.gauss(0, sigma * (hi - lo))
    return clip_genes(out)


def tournament(pop, fit, k=3):
    best = max(random.sample(range(len(pop)), k), key=lambda i: fit[i])
    return pop[best]


def save(best, fitness, generation):
    with open('best_genes.json', 'w') as f:
        json.dump({'generation': generation, 'fitness': fitness,
                   'names': [g[0] for g in GENE_SPEC], 'genes': best}, f, indent=2)


def main():
    pop_size = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    generations = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    elite = 2
    random.seed(0)

    # Start from the defaults AND from a "never thrust" individual, so the GA can only
    # improve on a ship that stays put and shoots (which survives well).
    stationary = [0.0 if name in ('rev_high', 'rev_low', 'fwd_low') else default
                  for (name, default, _lo, _hi) in GENE_SPEC]
    pop = [list(DEFAULT_GENES), stationary] + [random_near_default() for _ in range(pop_size - 2)]
    workers = max(1, (os.cpu_count() or 2) - 1)
    best, best_fit = None, -1e18

    try:
        with Pool(workers) as pool:
            for gen in range(generations):
                t0 = time.time()
                fit = pool.map(evaluate, pop)
                order = sorted(range(len(pop)), key=lambda i: fit[i], reverse=True)
                if fit[order[0]] > best_fit:
                    best, best_fit = list(pop[order[0]]), fit[order[0]]
                save(best, best_fit, gen)
                print(f'gen {gen + 1}/{generations}  best={best_fit:.2f}  '
                      f'mean={sum(fit) / len(fit):.2f}  ({time.time() - t0:.0f}s)', flush=True)

                nxt = [list(pop[i]) for i in order[:elite]]
                while len(nxt) < pop_size:
                    nxt.append(mutate(crossover(tournament(pop, fit), tournament(pop, fit))))
                pop = nxt
    except KeyboardInterrupt:
        print('\nstopped early, using best so far')

    if best is not None:
        print('\nPaste this into snoopy_flying_ace.py (replace the DEFAULT_GENES line):')
        print('DEFAULT_GENES = ' + repr([round(v, 4) for v in best]))


if __name__ == '__main__':
    main()
