# cognitive_modules/obstacle_evaluator.py
from .base_evaluator import BaseEvaluator
from misc import compute_distance_from_center, is_within_distance, dist


class StaticObstructionEvaluator(BaseEvaluator):
    """Sostituisce lo Scenario 3: Evitamento ostacoli statici"""

    def _detect_static_hazards(self, current_waypoint, element_query, angle_interval=[0, 90]):
        """Ex static_obstacle_manager: rileva coni e cartelli nell'ambiente"""
        sys = self.core_system
        obs_list = sys._world.get_actors().filter(element_query)
        obs_list = [o for o in obs_list if is_within_distance(o.get_transform(), sys._vehicle.get_transform(), 20,
                                                              angle_interval=angle_interval)]

        if not obs_list:
            return False, None, -1

        obs_list = sorted(obs_list, key=lambda x: dist(x, current_waypoint))

        if "constructioncone" in element_query:
            return True, obs_list[0], dist(obs_list[0], current_waypoint)

        return sys._vehicle_obstacle_detected(
            obs_list, max(sys._behavior.min_proximity_threshold, sys._speed_limit / 2), up_angle_th=60
        )

    def evaluate(self, **kwargs):
        current_waypoint = kwargs.get('ego_vehicle_wp')
        if not current_waypoint:
            return None

        sys = self.core_system

        # Scansione ambientale
        hazard_flag, hazard_obj, hazard_dist = self._detect_static_hazards(current_waypoint,
                                                                           "*static.prop.trafficwarning*")
        cone_flag, cone_obj, _ = self._detect_static_hazards(current_waypoint, "*static.prop.constructioncone*")

        # Salviamo i flag nel core_system per farli leggere al NavigationCruiseEvaluator
        sys.environmental_hazards = {'tw_state': hazard_flag, 'cone_state': cone_flag}

        if hazard_flag is True:
            actual_dist = compute_distance_from_center(actor1=sys._vehicle, actor2=hazard_obj, distance=hazard_dist)

            evasion_path = sys._bypass_engine.compute_evasion_trajectory(hazard_obj, current_waypoint, 1, 20, actual_dist, sys._speed_limit)

            if evasion_path:
                print("[Cognition] -> Trajectory alteration: bypassing environmental hazard.")
                sys._BehaviorAgent__update_global_plan(overtake_path=evasion_path)

            if not sys._bypass_engine.is_bypassing and actual_dist < 6:
                print("--- [Cognition] Bypass unfeasible. Executing critical halt.")
                return self.halt_vehicle()

        elif not hazard_flag and cone_flag and not sys._bypass_engine.is_bypassing:
            o_loc = cone_obj.get_location()
            o_wp = sys._map.get_waypoint(o_loc)

            if o_wp.lane_id == current_waypoint.lane_id:
                print("[Cognition] -> Debris detected in current lane. Applying negative lateral shift.")
                sys._local_planner.set_lateral_offset(
                    -2 * cone_obj.bounding_box.extent.y - sys._vehicle.bounding_box.extent.y)
            else:
                print("[Cognition] -> Debris detected in adjacent lane. Applying positive lateral shift.")
                sys._local_planner.set_lateral_offset(
                    .2 * cone_obj.bounding_box.extent.y + sys._vehicle.bounding_box.extent.y)

        return None