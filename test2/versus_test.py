# Head-to-head check for the "defeat test_controller.py" requirement:
# 5 games, unlimited bullets, 3 lives, both ships in the same game.
# Winner = most asteroids hit in total; ties broken by mean accuracy.
#
# Usage: python versus_test.py

import random

import numpy as np

from kesslergame import Scenario, KesslerGame, GraphicsType
try:
    from kesslergame import TrainerEnvironment
except ImportError:
    TrainerEnvironment = None

from test_controller import TestController
from snoopy_flying_ace import SnoopyFlyingAce

NUM_GAMES = 5
NUM_ASTEROIDS = 10
TIME_LIMIT = 60


def make_game():
    settings = {'perf_tracker': False,
                'graphics_type': GraphicsType.NoGraphics,
                'realtime_multiplier': 0,
                'graphics_obj': None,
                'frequency': 30}
    if TrainerEnvironment is not None:
        return TrainerEnvironment(settings=settings)
    return KesslerGame(settings=settings)


def main():
    mine = {'hits': 0, 'acc': [], 'deaths': 0}
    theirs = {'hits': 0, 'acc': [], 'deaths': 0}

    for game_no in range(NUM_GAMES):
        random.seed(100 + game_no)
        np.random.seed(100 + game_no)
        scenario = Scenario(name=f'versus_{game_no}',
                            num_asteroids=NUM_ASTEROIDS,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3},
                                {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, 'mines_remaining': 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=TIME_LIMIT,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)
        score, _ = make_game().run(scenario=scenario,
                                   controllers=[SnoopyFlyingAce(), TestController()])
        for side, team in ((mine, score.teams[0]), (theirs, score.teams[1])):
            side['hits'] += team.asteroids_hit
            side['acc'].append(team.accuracy)
            side['deaths'] += team.deaths
        print(f'game {game_no + 1}: SnoopyFlyingAce hits={score.teams[0].asteroids_hit} '
              f'acc={score.teams[0].accuracy:.2f} deaths={score.teams[0].deaths} | '
              f'TestController hits={score.teams[1].asteroids_hit} '
              f'acc={score.teams[1].accuracy:.2f} deaths={score.teams[1].deaths}', flush=True)

    my_acc = sum(mine['acc']) / NUM_GAMES
    their_acc = sum(theirs['acc']) / NUM_GAMES
    print(f"\nTOTAL  mine: hits={mine['hits']} mean acc={my_acc:.3f} deaths={mine['deaths']}")
    print(f"       test: hits={theirs['hits']} mean acc={their_acc:.3f} deaths={theirs['deaths']}")
    if mine['hits'] > theirs['hits'] or (mine['hits'] == theirs['hits'] and my_acc > their_acc):
        print('RESULT: you beat test_controller')
    else:
        print('RESULT: you did NOT beat test_controller yet')


if __name__ == '__main__':
    main()
