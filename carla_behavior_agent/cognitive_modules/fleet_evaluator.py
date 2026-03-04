# cognitive_modules/fleet_evaluator.py
from .base_evaluator import BaseEvaluator
from misc import compute_distance_from_center, get_speed


class FleetProximityEvaluator(BaseEvaluator):
    def evaluate(self, **kwargs):
        sys = self.core_system
        current_wp = kwargs.get('ego_vehicle_wp')
        debug_mode = kwargs.get('debug', False)

        # Scansione della flotta nei paraggi
        v_state, vehicle, v_dist = self.scan_for_fleet(current_wp)

        if not v_state:
            return None

        v_dist = compute_distance_from_center(actor1=sys._vehicle, actor2=vehicle, distance=v_dist)
        v_wp, is_parked = sys._parked_vehicle(vehicle)
        print(f"[Cognition] -> Fleet contact [{vehicle.id}] at {v_dist:.1f}m. Static: {is_parked}")

        if current_wp.lane_id == v_wp.lane_id and self.is_a_bicycle(vehicle.type_id):
            ego_yaw = abs(sys._vehicle.get_transform().rotation.yaw)
            v_yaw = abs(vehicle.get_transform().rotation.yaw)
            if self.is_road_straight(ego_yaw=ego_yaw, vehicle_yaw=v_yaw):
                if self.is_bicycle_near_center(vehicle_location=vehicle.get_location(),
                                          ego_vehicle_wp=current_wp) and get_speed(sys._vehicle) < 0.1:
                    print("--- [Cognition] Centered cyclist detected. Engaging bypass maneuver.")
                    bypass_path = sys._bypass_engine.compute_evasion_trajectory(
                        target_entity=vehicle,
                        current_wp=current_wp,
                        base_offset=1,
                        proximity_margin=v_dist,
                        max_velocity=sys._speed_limit
                    )
                    if bypass_path: sys._BehaviorAgent__update_global_plan(overtake_path=bypass_path)
                    if not sys._bypass_engine.is_bypassing: return self.halt_vehicle()
                else:
                    print("--- [Cognition] Cyclist offset. Applying lateral displacement.")
                    sys._local_planner.set_lateral_offset(
                        -(2.5 * vehicle.bounding_box.extent.y + sys._vehicle.bounding_box.extent.y))
                    return sys._BehaviorAgent__normal_behaviour(debug=debug_mode)
            elif get_speed(vehicle) < 1:
                sys.set_target_speed(sys._approach_speed)
                return sys._local_planner.run_step()
            else:
                return self.adaptive_cruise_control(vehicle, v_dist, debug=debug_mode)

        elif current_wp.lane_id != v_wp.lane_id and current_wp == -1:
            sys._local_planner.set_lateral_offset(
                .2 * vehicle.bounding_box.extent.y + sys._vehicle.bounding_box.extent.y)
            return sys._BehaviorAgent__normal_behaviour(debug=debug_mode)

        elif v_dist < sys._behavior.braking_distance and not sys._stuck:
            print("--- [Cognition] Critical fleet proximity. Executing halt.")
            sys._stuck = True
            return self.halt_vehicle()

        elif v_wp.lane_id == current_wp.lane_id and is_parked and not current_wp.is_junction:
            bypass_path = sys._bypass_engine.compute_evasion_trajectory(
                target_entity=vehicle,
                current_wp=current_wp,
                base_offset=1,
                proximity_margin=v_dist,
                max_velocity=sys._speed_limit
            )
            if bypass_path: sys._BehaviorAgent__update_global_plan(overtake_path=bypass_path)
            if not sys._bypass_engine.is_bypassing: return self.halt_vehicle()
            return sys._BehaviorAgent__normal_behaviour(debug=debug_mode)

        else:
            print("--- [Cognition] Adaptive cruise active. Tracking target.")
            return self.adaptive_cruise_control(vehicle, v_dist, debug=debug_mode)

    def is_road_straight(self, ego_yaw, vehicle_yaw, tolerance=10):
        """
        This function checks if the road is straight. In particular, it checks if the yaw of the ego vehicle
        and the vehicle in front are similar.

            :param ego_yaw (float): yaw of the ego vehicle.
            :param vehicle_yaw (float): yaw of the vehicle in front.
            :param tolerance (int): tolerance value to check if the road is straight.

            :return (bool): True if the road is straight, False otherwise.
        """
        return abs(ego_yaw - vehicle_yaw) < tolerance


    def is_bicycle_near_center(self,vehicle_location, ego_vehicle_wp):
        """
        This function checks if the bicycle is near the center of the lane.

            :param vehicle_location (carla.Location): location of the vehicle.
            :param ego_vehicle_wp (carla.Waypoint): waypoint of the ego vehicle.

            :return (bool): True if the bicycle is near the center of the lane, False otherwise.
        """
        lane_center_offset = 0.3  # How close to the center the bicycle needs to be considered in the center
        vehicle_y = vehicle_location.y
        lane_center_y = ego_vehicle_wp.transform.location.y
        return abs(vehicle_y - lane_center_y) < lane_center_offset