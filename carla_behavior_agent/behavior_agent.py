# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from tactical_modules.trajectory_bypass_engine import TrajectoryBypassEngine
from tactical_modules.intersection_navigation_engine import IntersectionNavigationEngine, IntersectionTopology
from cognitive_modules.signal_evaluator import TrafficSignalEvaluator
from cognitive_modules.pedestrian_evaluator import BipedalHazardEvaluator
from cognitive_modules.obstacle_evaluator import StaticObstructionEvaluator
from cognitive_modules.stop_sign_evaluator import MandatoryStopEvaluator
from cognitive_modules.fleet_evaluator import FleetProximityEvaluator
from cognitive_modules.cruise_evaluator import NavigationCruiseEvaluator

from misc import *
# from utils import configure_logger

# logger = configure_logger()

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='cautious', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)

        # Vehicle information
        self._approach_speed = 10.0                                 # Approach speed of the agent
        self._behavior = \
            Aggressive() if behavior == 'aggressive' else \
            Normal() if behavior == 'normal' else \
            Cautious()            
        self._direction = None                                      # Current direction of the agent
        self._incoming_direction = None                             # Incoming direction of the agent
        self._incoming_waypoint = None                              # Incoming waypoint of the agent
        self._look_ahead_steps = 0                                  # Number of steps to look ahead
        self._min_speed = 5                                         # Minimum speed of the agent
        self._speed = 0                                             # Current speed of the agent
        self._speed_limit = 0                                       # Speed limit of the agent
        self._stuck = False                                         # Flag indicating if the agent is stuck

        self._bypass_engine = TrajectoryBypassEngine(self._vehicle, opt_dict)
        self._evaluators_chain = [
            TrafficSignalEvaluator(core_system=self),
            BipedalHazardEvaluator(core_system=self),
            StaticObstructionEvaluator(core_system=self),
            MandatoryStopEvaluator(core_system=self),
            FleetProximityEvaluator(core_system=self),
            NavigationCruiseEvaluator(core_system=self)
        ]

        self._bypass_engine = TrajectoryBypassEngine(self._vehicle, opt_dict)
        self._navigation_engine = IntersectionNavigationEngine(self._vehicle, opt_dict)
        
        # Parameters for agent behavior
        self._behavior = Cautious() if behavior == 'cautious' else Aggressive() if behavior == 'aggressive' else Normal()
        self._is_raining = False

    def run_step(self, debug=False):
        """Metodo di ciclo principale riprogettato ad architettura modulare"""
        self.__update_information(debug=debug)

        # Data-wrapper minimale per i moduli decisionali
        environment_context = {
            'ego_vehicle_wp': self._map.get_waypoint(self._vehicle.get_location()),
            'debug': debug
        }

        # Motore cognitivo: attraversa la Chain of Responsibility
        for evaluator in self._evaluators_chain:
            action = evaluator.evaluate(**environment_context)
            if action is not None:
                return action

        return carla.VehicleControl()


    def __normal_behaviour(self, debug = False):
        """
        This method is in charge of the normal behavior of the agent. In particular, it is in charge of setting the speed of the agent and
        running the local planner.
        
            :param debug (bool): debug flag to print information.
        """
        # Set the speed of the agent.
        target_speed = min(
            [self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]
        )
        self._local_planner.set_speed(target_speed)
        control = self._local_planner.run_step(debug=debug)
        return control
    
    def __emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        # Get the vehicle control
        control = carla.VehicleControl()
        # Set the throttle and brake values to 0.0 and self._max_brake, respectively.
        control.throttle = 0.0
        control.brake = self._max_brake
        # Set the hand brake to False.
        control.hand_brake = False
        return control
    
    def __update_global_plan(self, overtake_path : list) -> None:
        """Ex update_global_plan: Nasconde la logica di forzatura rotta locale"""
        new_plan = self._local_planner.set_overtake_plan(
            overtake_plan=overtake_path,
            overtake_distance=self._bypass_engine.required_clearance
        )
        self.set_target_speed(2 * self._speed_limit)
        self.set_global_plan(new_plan)
    
    def __update_information(self, debug = False):
        """
        This method updates the information regarding the ego vehicle based on the surrounding world.
        
            :param debug (bool): debug flag to print information.
        """
        # Get the current weather of the simulation.
        self._weather = self._world.get_weather()
        precipitation_intensity = self._weather.precipitation
        preciptitation_deposits = self._weather.precipitation_deposits
        self._is_raining = precipitation_intensity > 50 or preciptitation_deposits > 55
                         
        # Update the speed of the agent.
        self._speed = get_speed(self._vehicle)

        # Update the speed limit of the agent.
        self._speed_limit = self._vehicle.get_speed_limit() 
        self._speed_limit -= 5 if self._is_raining else 0

        # Update the local planner speed.
        self._local_planner.set_speed(self._speed_limit)

        # Update the direction of the agent.
        self._direction = self._local_planner.target_road_option if self._local_planner.target_road_option is not None else RoadOption.LANEFOLLOW

        # Update the look ahead steps of the agent.
        self._look_ahead_steps = int(self._speed_limit / 10)

        # Update the incoming waypoint and direction of the agent.
        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps
        )

        # Update the incoming direction of the agent.
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW
            
        # Update the behavior of the agent.
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1       


        if self._bypass_engine.evasion_lock > 0:
            self._bypass_engine.evasion_lock -= 1
        else:
            self._bypass_engine.is_bypassing = False

        # Gestione dei frame di blocco per l'attraversamento incroci
        if self._navigation_engine.is_traversing and self._navigation_engine.active_traversal_frames <= 0:
            self._navigation_engine.is_traversing = False
            self._navigation_engine.active_traversal_frames = 0
            self._navigation_engine.halt_frames = 0
        elif self._navigation_engine.active_traversal_frames > 0:
            self._navigation_engine.active_traversal_frames -= 1

        if self._navigation_engine.starboard_turn_frames > 0:
            self._navigation_engine.starboard_turn_frames -= 1
        
        if self._is_raining:
            print("[WORLD] It is raining!")
        print("[VEHICLE] Vehicle Speed: {0} - Speed Limit: {1}".format(self._speed, self._speed_limit))
