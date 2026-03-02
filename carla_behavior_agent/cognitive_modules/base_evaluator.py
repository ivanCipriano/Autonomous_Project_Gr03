# cognitive_modules/base_evaluator.py
import carla
import numpy as np
from misc import is_within_distance, is_a_bicycle, dist, get_speed, positive
from local_planner import RoadOption


class BaseEvaluator:
    """Interfaccia base per la catena di valutazione cognitiva e libreria di funzioni percettive."""

    def __init__(self, core_system):
        self.core_system = core_system

    def evaluate(self, **kwargs) -> carla.VehicleControl:
        raise NotImplementedError("Il metodo evaluate deve essere implementato.")

    def halt_vehicle(self) -> carla.VehicleControl:
        cmd = carla.VehicleControl()
        cmd.throttle = 0.0
        cmd.brake = self.core_system._max_brake
        cmd.hand_brake = False
        return cmd

    def adaptive_cruise_control(self, target_vehicle, distance, debug=False):
        """Ex car_following_manager: Modula la velocità in base al veicolo che precede."""
        sys = self.core_system
        vehicle_speed = get_speed(target_vehicle)
        delta_v = max(1, (sys._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        if sys._behavior.safety_time > ttc > 0.0:
            target_speed = min([positive(vehicle_speed - sys._behavior.speed_decrease), sys._behavior.max_speed,
                                sys._speed_limit - sys._behavior.speed_lim_dist])
        elif 2 * sys._behavior.safety_time > ttc >= sys._behavior.safety_time:
            target_speed = min([max(sys._min_speed, vehicle_speed), sys._behavior.max_speed,
                                sys._speed_limit - sys._behavior.speed_lim_dist])
        else:
            target_speed = min([sys._behavior.max_speed, sys._speed_limit - sys._behavior.speed_lim_dist])

        sys._local_planner.set_speed(target_speed)
        return sys._local_planner.run_step(debug=debug)

    def _process_tailgating(self, waypoint, vehicle_list):
        """Ex _tailgating: Gestisce le dinamiche di inseguimento e cambio corsia."""
        sys = self.core_system
        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change
        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()
        speed_limit = sys._vehicle.get_speed_limit()

        behind_v_state, behind_v, _ = sys._vehicle_obstacle_detected(
            vehicle_list, max(sys._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180,
            low_angle_th=160)

        if behind_v_state and sys._speed < get_speed(behind_v):
            if (
                    right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_v_state, _, _ = sys._vehicle_obstacle_detected(vehicle_list,
                                                                   max(sys._behavior.min_proximity_threshold,
                                                                       speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_v_state:
                    print("[Cognition] -> Engaging right evasion to clear lane.")
                    sys._behavior.tailgate_counter = 200
                    sys.set_destination(sys._local_planner.target_waypoint.transform.location,
                                        right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_v_state, _, _ = sys._vehicle_obstacle_detected(vehicle_list,
                                                                   max(sys._behavior.min_proximity_threshold,
                                                                       speed_limit / 2), up_angle_th=180,
                                                                   lane_offset=-1)
                if not new_v_state:
                    print("[Cognition] -> Engaging left evasion to clear lane.")
                    sys._behavior.tailgate_counter = 200
                    sys.set_destination(sys._local_planner.target_waypoint.transform.location,
                                        left_wpt.transform.location)

    def scan_for_fleet(self, waypoint):
        """Ex collision_and_car_avoid_manager: Rileva veicoli e bici nei dintorni."""
        sys = self.core_system
        v_list = sys._world.get_actors().filter("*vehicle*")
        v_list = [v for v in v_list if dist(v, waypoint) < 13 and v.id != sys._vehicle.id]

        if not v_list:
            return False, None, -1

        bicycle_list = [b for b in v_list if
                        is_a_bicycle(b.type_id) and is_within_distance(b.get_transform(), sys._vehicle.get_transform(),
                                                                       10, angle_interval=[0, 90])]
        if len(bicycle_list) == 1:
            print('[Cognition] -> Cyclist track intercepted.')
            return True, bicycle_list[0], dist(bicycle_list[0], waypoint)

        speed_limit = sys._vehicle.get_speed_limit()

        if sys._direction == RoadOption.CHANGELANELEFT:
            v_state, v_obj, v_dist = sys._vehicle_obstacle_detected(v_list, max(sys._behavior.min_proximity_threshold,
                                                                                speed_limit / 2), up_angle_th=180,
                                                                    lane_offset=-1)
        elif sys._direction == RoadOption.CHANGELANERIGHT:
            v_state, v_obj, v_dist = sys._vehicle_obstacle_detected(v_list, max(sys._behavior.min_proximity_threshold,
                                                                                speed_limit / 2), up_angle_th=180,
                                                                    lane_offset=1)
        else:
            v_state, v_obj, v_dist = sys._vehicle_obstacle_detected(v_list, max(sys._behavior.min_proximity_threshold,
                                                                                speed_limit / 3), up_angle_th=30)
            if v_state:
                v_wp = sys._map.get_waypoint(v_obj.get_location())
                if v_wp.is_junction:
                    return v_state, v_obj, v_dist
                proj_lane = v_wp.get_left_lane()
                ego_wp = sys._map.get_waypoint(sys._vehicle.get_location())
                if proj_lane and proj_lane.lane_type == carla.LaneType.Driving and proj_lane.get_left_lane().lane_id == ego_wp.lane_id:
                    if v_obj.get_location().distance(
                            proj_lane.get_left_lane().transform.location) > sys._vehicle.bounding_box.extent.y + v_obj.bounding_box.extent.y:
                        return False, None, -1
            if not v_state and sys._direction == RoadOption.LANEFOLLOW and not waypoint.is_junction and sys._speed > 10 and sys._behavior.tailgate_counter == 0:
                self._process_tailgating(waypoint, v_list)

        return v_state, v_obj, v_dist