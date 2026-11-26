# Adapted from scott_dick_controller.py, Fall 2023, Dr. Scott Dick

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

class SnoopyFlyingAce(KesslerController):
    
    def __init__(self):
        self.initial_asteroids_logged = False
        self.eval_frames = 0 # What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.002), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)
        
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
     
        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()
        
        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        # self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        # self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        # self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)

        # FUZZY MOVEMENT CONTROL SYSTEM

        # Declare fuzzy input and output variables
        asteroid_distance = ctrl.Antecedent(np.arange(0, 801, 1),'asteroid_distance')
        ship_speed = ctrl.Antecedent(np.arange(0, 401, 1),'ship_speed')
        ship_thrust = ctrl.Consequent(np.arange(0, 20.1, 0.1),'ship_thrust')

        # MEMBERSHIP FUNCTIONS

        # Asteroid distance: Near, Medium, Far
        asteroid_distance['Near'] = fuzz.trimf(asteroid_distance.universe, [0, 0, 200])
        asteroid_distance['Medium'] = fuzz.trimf(asteroid_distance.universe, [100, 300, 500])
        asteroid_distance['Far'] = fuzz.trimf(asteroid_distance.universe, [350, 800, 800])

        # Ship speed: Slow, Medium, Fast
        ship_speed['Slow'] = fuzz.trimf(ship_speed.universe, [0, 0, 100])
        ship_speed['Medium'] = fuzz.trimf(ship_speed.universe, [50, 150, 250])
        ship_speed['Fast'] = fuzz.trimf(ship_speed.universe, [200, 400, 400])

        # Ship thrust: Zero, Low, Medium, High
        ship_thrust['Zero'] = fuzz.trimf(ship_thrust.universe, [0, 0, 5])
        ship_thrust['Low'] = fuzz.trimf(ship_thrust.universe, [0, 5, 10])
        ship_thrust['Medium'] = fuzz.trimf(ship_thrust.universe, [5, 10, 15])
        ship_thrust['High'] = fuzz.trimf(ship_thrust.universe, [10, 15, 20])

        # FUZZY RULES

        # Near asteroids: use less thrust
        thrust_rule1 = ctrl.Rule(asteroid_distance['Near'] & ship_speed['Slow'],ship_thrust['Zero'])
        thrust_rule2 = ctrl.Rule(asteroid_distance['Near'] & ship_speed['Medium'],ship_thrust['Zero'])
        thrust_rule3 = ctrl.Rule(asteroid_distance['Near'] & ship_speed['Fast'],ship_thrust['Zero'])

        # Medium distance: moderate thrust
        thrust_rule4 = ctrl.Rule(asteroid_distance['Medium'] & ship_speed['Slow'],ship_thrust['Medium'])
        thrust_rule5 = ctrl.Rule(asteroid_distance['Medium'] & ship_speed['Medium'],ship_thrust['Low'])
        thrust_rule6 = ctrl.Rule(asteroid_distance['Medium'] & ship_speed['Fast'],ship_thrust['Zero'])

        # Far away: allow more thrust

        thrust_rule7 = ctrl.Rule(asteroid_distance['Far'] & ship_speed['Slow'],ship_thrust['High'])
        thrust_rule8 = ctrl.Rule(asteroid_distance['Far'] & ship_speed['Medium'],ship_thrust['Medium'])
        thrust_rule9 = ctrl.Rule(asteroid_distance['Far'] & ship_speed['Fast'],ship_thrust['Low'])

        # CREATE FUZZY CONTROL SYSTEM
        self.movement_control = ctrl.ControlSystem([
            thrust_rule1,
            thrust_rule2,
            thrust_rule3,
            thrust_rule4,
            thrust_rule5,
            thrust_rule6,
            thrust_rule7,
            thrust_rule8,
            thrust_rule9
        ])

        # FUZZY MINE CONTROL SYSTEM

        mine_distance = ctrl.Antecedent(np.arange(0, 801, 1),'mine_distance')
        mine_speed = ctrl.Antecedent(np.arange(0, 41, 1),'mine_speed')
        mine_output = ctrl.Consequent(np.arange(0, 1.01, 0.01),'mine_output')

        # Distance membership functions
        mine_distance['Near'] = fuzz.trimf(mine_distance.universe, [0, 0, 80])
        mine_distance['Far'] = fuzz.trimf(mine_distance.universe, [40, 800, 800])

        # Speed membership functions
        mine_speed['Slow'] = fuzz.trimf(mine_speed.universe, [0, 0, 10])
        mine_speed['Fast'] = fuzz.trimf(mine_speed.universe, [5, 40, 40])

        # Mine output membership functions
        mine_output['No'] = fuzz.trimf(mine_output.universe, [0, 0, 0.5])
        mine_output['Yes'] = fuzz.trimf(mine_output.universe, [0.5, 1, 1])

        # Fuzzy rules
        mine_rule1 = ctrl.Rule(mine_distance['Near'] & mine_speed['Slow'],mine_output['No'])
        mine_rule2 = ctrl.Rule(mine_distance['Near'] & mine_speed['Fast'],mine_output['Yes'])
        mine_rule3 = ctrl.Rule(mine_distance['Far'],mine_output['No'])

        self.mine_control = ctrl.ControlSystem([
            mine_rule1,
            mine_rule2,
            mine_rule3
        ])

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.


        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        
        # EVALUATE FUZZY MOVEMENT CONTROL SYSTEM

        # Calculate the ship's current speed
        ship_speed_value = math.sqrt(
            ship_state["velocity"][0]**2 +
            ship_state["velocity"][1]**2
        )

        # Get distance to the nearest asteroid
        if closest_asteroid is not None:
            asteroid_distance_value = closest_asteroid["dist"]
        else:
            asteroid_distance_value = 800.0

        # Keep inputs within the fuzzy universes
        asteroid_distance_value = np.clip(
            asteroid_distance_value, 0, 800
        )

        ship_speed_value = np.clip(
            ship_speed_value, 0, 400
        )

        # Create fuzzy simulation
        movement = ctrl.ControlSystemSimulation(
            self.movement_control,
            flush_after_run=1
        )

        # Provide input values
        movement.input['asteroid_distance'] = asteroid_distance_value
        movement.input['ship_speed'] = ship_speed_value

        # Evaluate fuzzy rules
        movement.compute()

        # Retrieve fuzzy output
        thrust = movement.output['ship_thrust']

        # EVALUATE FUZZY MINE CONTROL SYSTEM

        drop_mine = False

        if closest_asteroid is not None:

            mine = ctrl.ControlSystemSimulation(
                self.mine_control,
                flush_after_run=1
            )

            mine.input['mine_distance'] = np.clip(
                closest_asteroid["dist"], 0, 800
            )

            mine.input['mine_speed'] = np.clip(
                ship_speed_value, 0, 40
            )

            mine.compute()

            # Only drop a mine if the fuzzy output says Yes and the ship has mines remaining
            if ship_state["mines_remaining"] > 0:
                drop_mine = bool(mine.output['mine_output'] >= 0.5)

        # Handle the case where there are no asteroids
        if closest_asteroid is None:
            self.eval_frames += 1
            return thrust, 0.0, False, False


        # closest_asteroid is now the nearest asteroid object
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction)
        # Based on Law of Cosines calculation, see notes
        
        # Determinant of the quadratic formula b^2-4ac
                # Calculate relative position and velocity
        asteroid_x = closest_asteroid["aster"]["position"][0]
        asteroid_y = closest_asteroid["aster"]["position"][1]

        asteroid_vx = closest_asteroid["aster"]["velocity"][0]
        asteroid_vy = closest_asteroid["aster"]["velocity"][1]

        relative_x = asteroid_x - ship_pos_x
        relative_y = asteroid_y - ship_pos_y

        bullet_speed = 800.0

        # Solve:
        # |relative_position + asteroid_velocity * t| = bullet_speed * t
        a = asteroid_vx**2 + asteroid_vy**2 - bullet_speed**2
        b = 2 * (relative_x * asteroid_vx + relative_y * asteroid_vy)
        c = relative_x**2 + relative_y**2

        bullet_t = 0.0

        if abs(a) > 1e-9:
            discriminant = b**2 - 4 * a * c

            if discriminant >= 0:
                root = math.sqrt(discriminant)

                roots = [
                    (-b - root) / (2 * a),
                    (-b + root) / (2 * a)
                ]

                positive_roots = [t for t in roots if t > 0]

                if positive_roots:
                    bullet_t = min(positive_roots)

        elif abs(b) > 1e-9:
            # Linear fallback if the quadratic coefficient is approximately zero
            possible_time = -c / b

            if possible_time > 0:
                bullet_t = possible_time

        # Predict the asteroid's position when the bullet arrives
        # Include one game frame of delay, as in the original controller
        intrcpt_x = asteroid_x + asteroid_vx * (bullet_t + 1/30)
        intrcpt_y = asteroid_y + asteroid_vy * (bullet_t + 1/30)

        my_theta1 = math.atan2(intrcpt_y - ship_pos_y, intrcpt_x - ship_pos_x)

        shooting_theta = my_theta1 - math.radians(ship_state["heading"])

        # Wrap the angle to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Keep fuzzy inputs inside their defined universes
        bullet_t = np.clip(bullet_t, 0.0, 0.998)
        shooting_theta = np.clip(
            shooting_theta,
            -math.pi / 30,
            math.pi / 30 - 0.002
        )
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
        self.eval_frames +=1
        
        #DEBUG
        # print(
        #     "Distance:", asteroid_distance_value,
        #     "Speed:", ship_speed_value,
        #     "Thrust:", thrust,
        #     "Turn:", turn_rate,
        #     "Fire:", fire
        # )

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "SnoopyFlyingAce"