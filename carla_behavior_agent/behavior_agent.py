# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import math
import sys
import random
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal

from misc import *
from overtake_manager import OvertakeManager

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.
        """
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self._overtake_manager = OvertakeManager(self._vehicle, self._world, self._map, self._global_planner)

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()
        elif behavior == 'normal':
            self._behavior = Normal()
        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)
        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.
        """
        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v):
            return v.get_location().distance(waypoint.transform.location)

        # -------------------------------------------------------------
        # 1. RILEVAMENTO BICI (Soglia a 25m per scarto tempestivo)
        # -------------------------------------------------------------
        bicycle_list = [b for b in vehicle_list if is_a_bicycle(b.type_id) and 
                        is_within_distance(b.get_transform(), self._vehicle.get_transform(), 25, angle_interval=[0, 90])
                        and b.id != self._vehicle.id]
        
        if bicycle_list:
            closest_bike = sorted(bicycle_list, key=dist)[0]
            return True, closest_bike, dist(closest_bike)

        # -------------------------------------------------------------
        # 2. RILEVAMENTO AUTO STANDARD (Orizzonte aumentato a 45 metri!)
        # -------------------------------------------------------------
        # 45 metri garantiscono al sistema lo spazio fisico per frenare da 50+ km/h
        car_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id and not is_a_bicycle(v.type_id)]

        if not car_list:
            return False, None, -1

        # Usiamo self._speed_limit come raggio visivo profondo al posto di limitarlo a /3
        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                car_list, max(self._behavior.min_proximity_threshold, self._speed_limit), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                car_list, max(self._behavior.min_proximity_threshold, self._speed_limit), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                car_list, max(self._behavior.min_proximity_threshold, self._speed_limit), up_angle_th=30)

            # Controllo anti-tailgating per cambiare corsia se siamo bloccati dietro a un'auto lenta
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, car_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision with any pedestrian.
        """
        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        
        walker_list = [w for w in walker_list if dist(w) < 15]

        # -------------------------------------------------------------
        # FILTRO INTELLIGENTE: ELIMINA I CICLISTI/MOTOCICLISTI
        # -------------------------------------------------------------
        valid_walkers = []
        for w in walker_list:
            w_loc = w.get_location()
            is_rider = False
            for v in vehicle_list:
                v_loc = v.get_location()
                if math.hypot(w_loc.x - v_loc.x, w_loc.y - v_loc.y) < 2.0:
                    is_rider = True
                    break
            
            if not is_rider:
                valid_walkers.append(w)
                
        if not valid_walkers:
            return False, None, -1
        # -------------------------------------------------------------

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(valid_walkers, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(valid_walkers, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(valid_walkers, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's someone in front of us.
        """
        import sys
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Se il Time To Collision (TTC) è sotto la soglia di sicurezza, rallenta gradualmente
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            
            print(f"[CAR FOLLOWING] Auto davanti a {distance:.1f}m. Rallento dolcemente a {target_speed:.1f} km/h.")
            sys.stdout.flush()
            
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Mantenimento della distanza
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            
            print(f"[CAR FOLLOWING] In accodamento. Mantengo la velocità a {target_speed:.1f} km/h.")
            sys.stdout.flush()
            
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Condotta normale se la distanza è enorme
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def static_obstacle_manager(self, waypoint, static_element="*static.prop*", angle_interval=[0, 90]):
        """
        Rileva ostacoli statici davanti al veicolo.
        """
        obstacles_list = self._world.get_actors().filter(static_element)
        search_distance = 45.0

        obstacles_list = [o for o in obstacles_list if is_within_distance(
            o.get_transform(), self._vehicle.get_transform(), search_distance, angle_interval=angle_interval)]

        if not obstacles_list:
            return False, None, -1

        def dist(o): return o.get_location().distance(waypoint.transform.location)

        obstacles_list = sorted(obstacles_list, key=dist)
        closest_obstacle = obstacles_list[0]
        
        return True, closest_obstacle, dist(closest_obstacle)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        """
        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            print(f"[PEDONE] Rilevato a {distance:.1f}m. Valuto la distanza di sicurezza.")
            sys.stdout.flush()

            if distance < self._behavior.braking_distance:
                print("[PEDONE] Troppo vicino! Freno d'emergenza.")
                sys.stdout.flush()
                return self.emergency_stop()
            else:
                print("[PEDONE] Distanza di sicurezza ok. Rallento l'approccio a 10 km/h.")
                sys.stdout.flush()
                self._local_planner.set_speed(10.0)
                return self._local_planner.run_step(debug=debug)

        # ---------------------------------------------------------
        # SCENARIO AVANZATO: Gestione Ostacoli e Sorpasso
        # ---------------------------------------------------------
        o_state, obstacle, o_distance = self.static_obstacle_manager(ego_vehicle_wp, static_element="*static.prop.trafficwarning*")
        c_state, cone, c_distance = self.static_obstacle_manager(ego_vehicle_wp, static_element="*static.prop.constructioncone*")
        
        current_plan = self._local_planner.get_plan()

        if o_state and not self._overtake_manager.in_overtake and len(current_plan) > 0:
            obs_extent = max(obstacle.bounding_box.extent.y, obstacle.bounding_box.extent.x) if hasattr(obstacle, 'bounding_box') else 1.0
            ego_extent = max(self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            real_distance = o_distance - obs_extent - ego_extent

            end_destination_loc = current_plan[-1][0].transform.location
            overtake_path = self._overtake_manager.get_overtake_path(ego_vehicle_wp, o_distance, end_destination_loc)

            if overtake_path is not None:
                print(f"[OSTACOLO] Ostacolo a {real_distance:.1f}m. Inizio manovra di sorpasso.")
                sys.stdout.flush()
                self._local_planner.set_global_plan(overtake_path, clean_queue=True)
                self._overtake_manager.in_overtake = True
                self._local_planner.set_speed(25.0)
                return self._local_planner.run_step(debug=debug)
            else:
                if real_distance < 8.0:
                    return self.emergency_stop()
                else:
                    self._local_planner.set_speed(10.0)
                    return self._local_planner.run_step(debug=debug)

        elif self._overtake_manager.in_overtake:
            self._local_planner.set_speed(25.0)
            if not o_state and ego_vehicle_wp.lane_id == self._overtake_manager.original_lane_id:
                print("[SORPASSO] Rientro completato.")
                sys.stdout.flush()
                self._overtake_manager.in_overtake = False
                self._local_planner.set_lateral_offset(0.0)
        
        elif c_state and not self._overtake_manager.in_overtake:
            cone_wp = self._map.get_waypoint(cone.get_location())
            cone_extent_y = cone.bounding_box.extent.y if hasattr(cone, 'bounding_box') else 0.5
            ego_extent_y = self._vehicle.bounding_box.extent.y

            if cone_wp.lane_id == ego_vehicle_wp.lane_id:
                self._local_planner.set_lateral_offset(-(1.5 * cone_extent_y + ego_extent_y))
            else:
                dist_fisica = self._vehicle.get_location().distance(cone.get_location())
                if dist_fisica < 3.0:
                    self._local_planner.set_lateral_offset(0.5 * cone_extent_y + ego_extent_y)
                else:
                    self._local_planner.set_lateral_offset(0.0)

        # ---------------------------------------------------------
        # 2.2: Car & Bicycle behaviors (Fix Auto in Coda)
        # ---------------------------------------------------------
        vehicle_state, vehicle, vehicle_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            real_distance = vehicle_distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            
            vehicle_wp = self._map.get_waypoint(vehicle.get_location())
            parked = get_speed(vehicle) < 0.1
            
            # SCENARIO BICI
            if is_a_bicycle(vehicle.type_id):
                print(f"[BICICLETTA] Ciclista davanti a noi a {real_distance:.1f}m. Scarto laterale forzato.")
                sys.stdout.flush()
                
                offset_value = -(0.5 * vehicle.bounding_box.extent.y + self._vehicle.bounding_box.extent.y)
                if offset_value > -1.5:
                    offset_value = -1.5
                    
                self._local_planner.set_lateral_offset(offset_value)
                
                target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                return self._local_planner.run_step(debug=debug)

            # SCENARIO EMERGENZA PRIORITARIO (Controlla questo PRIMA di provare sorpassi!)
            elif real_distance < self._behavior.braking_distance:
                print(f"[AUTO] Veicolo troppo vicino ({real_distance:.1f}m)! Freno d'emergenza.")
                sys.stdout.flush()
                return self.emergency_stop()
            
            # SCENARIO AUTO PARCHEGGIATA O IN CODA (Se siamo a distanza di sicurezza)
            elif vehicle_wp.lane_id == ego_vehicle_wp.lane_id and parked and not ego_vehicle_wp.is_junction:               
                print(f"[AUTO] Veicolo in coda/parcheggiato a {real_distance:.1f}m. Tento il sorpasso.")
                sys.stdout.flush()
                
                if len(current_plan) > 0:
                    overtake_path = self._overtake_manager.get_overtake_path(ego_vehicle_wp, real_distance, current_plan[-1][0].transform.location)
                    if overtake_path:
                        print("[AUTO] Percorso libero! Inizio sorpasso.")
                        sys.stdout.flush()
                        self._local_planner.set_global_plan(overtake_path, clean_queue=True)
                        self._overtake_manager.in_overtake = True
                        self._local_planner.set_speed(25.0)
                        return self._local_planner.run_step(debug=debug)
                
                # Se NON ci sono le condizioni per superare la macchina, passiamo il veicolo al car_following_manager
                # Il controller rallenterà dolcemente l'auto fino a 0 km/h mettendola in coda in totale sicurezza!
                if not self._overtake_manager.in_overtake:
                    print("[AUTO] Impossibile sorpassare. Rallento per accodarmi.")
                    sys.stdout.flush()
                    return self.car_following_manager(vehicle, real_distance, debug=debug)
            
            # SCENARIO AUTO IN MOVIMENTO
            else:
                return self.car_following_manager(vehicle, real_distance, debug=debug)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([self._behavior.max_speed, self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            return self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            if not o_state and not c_state and not self._overtake_manager.in_overtake:
                self._local_planner.set_lateral_offset(0.0)
                
            target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            return self._local_planner.run_step(debug=debug)

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control