# tactical_modules/trajectory_bypass_engine.py
import carla
import math
from basic_agent import BasicAgent
from misc import get_distance, is_within_distance, compute_distance_from_center, get_speed


class TrajectoryBypassEngine(BasicAgent):
    """
    Motore tattico per la generazione di traiettorie di elusione e sorpasso.
    (Sostituisce ex OvertakeManager)
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._evasion_lock_frames = 0
        self._is_executing_bypass = False
        self._required_clearance = 0

    def compute_evasion_trajectory(self, target_entity: carla.Actor, current_wp: carla.Waypoint,
                                   base_offset: float = 1, opposite_offset: float = 0,
                                   proximity_margin: float = 18, max_velocity: float = 50):
        """Genera il tracciato per bypassare un ostacolo (ex run_step)"""

        if not opposite_offset:
            opposite_offset = self._estimate_opposite_clearance(target_entity, 30)

        v_length = self._vehicle.bounding_box.extent.x
        l_width = current_wp.lane_width

        self._required_clearance, hypotenuse = self._calculate_spatial_clearance(v_length, l_width, base_offset,
                                                                                 opposite_offset, proximity_margin)
        maneuver_time = self._estimate_maneuver_duration(self._vehicle, self._required_clearance)

        oncoming_travel_dist = maneuver_time * max_velocity / 3.6
        scan_radius = self._required_clearance + oncoming_travel_dist

        oncoming_entity = self._detect_oncoming_traffic(current_wp, scan_radius)

        next_ego_wp = current_wp.next(self._required_clearance)
        collision_risk = False

        try:
            next_ego_wp = next_ego_wp[0]
            if oncoming_entity:
                next_oncoming_wp = self._map.get_waypoint(oncoming_entity.get_location()).next(oncoming_travel_dist)[0]
                collision_risk = not is_within_distance(target_transform=next_oncoming_wp.transform,
                                                        reference_transform=next_ego_wp.transform, max_distance=30,
                                                        angle_interval=[0, 90])
        except Exception:
            collision_risk = False

        wp_in_junction = False if type(next_ego_wp) is list or not next_ego_wp else next_ego_wp.is_junction

        if not self._evasion_lock_frames and not (collision_risk or wp_in_junction):
            bypass_path = self._generate_lane_change_path(
                waypoint=current_wp, direction='left', distance_same_lane=base_offset,
                distance_other_lane=opposite_offset, lane_change_distance=hypotenuse,
                check=False, step_distance=self._sampling_resolution
            )
            if not bypass_path: return None

            self._evasion_lock_frames = int(round(maneuver_time) / self._world.get_snapshot().timestamp.delta_seconds)
            self._is_executing_bypass = True
            return bypass_path

    @property
    def is_bypassing(self):
        return self._is_executing_bypass

    @is_bypassing.setter
    def is_bypassing(self, val):
        self._is_executing_bypass = val

    @property
    def evasion_lock(self):
        return self._evasion_lock_frames

    @evasion_lock.setter
    def evasion_lock(self, val):
        self._evasion_lock_frames = val

    @property
    def required_clearance(self):
        return self._required_clearance

    # --- Metodi Privati Rinominati ---
    def _estimate_opposite_clearance(self, actor: carla.Actor, max_distance: float = 30) -> float:
        actor_length = actor.bounding_box.extent.x
        distance_other_lane = actor_length

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        target_lane_id = self._map.get_waypoint(actor.get_location()).lane_id

        filtered_list = [
            v for v in vehicle_list
            if v.id != actor.id and v.id != self._vehicle.id and get_distance(v, actor) < max_distance
               and self._map.get_waypoint(v.get_location()).lane_id == target_lane_id
        ]

        parked_list = [v for v in filtered_list if self._parked_vehicle(v)[1] == True]

        previous_vehicle = actor
        for v in parked_list:
            if is_within_distance(target_transform=v.get_transform(),
                                  reference_transform=previous_vehicle.get_transform(), max_distance=max_distance,
                                  angle_interval=[0, 60]):
                v_distance = compute_distance_from_center(actor1=previous_vehicle, actor2=v,
                                                          distance=get_distance(v, previous_vehicle))
            else:
                continue
            distance_other_lane += v.bounding_box.extent.x + v_distance
            previous_vehicle = v

        return distance_other_lane + 3

    def _detect_oncoming_traffic(self, ego_wp: carla.Waypoint, search_distance: float = 30):
        def _extend_bounding_box(actor):
            wp = self._map.get_waypoint(actor.get_location())
            transform = wp.transform
            forward_vector = transform.get_forward_vector()
            extent = actor.bounding_box.extent.x
            transform.location += carla.Location(x=extent * forward_vector.x, y=extent * forward_vector.y)
            return transform

        vehicle_list = self._get_ordered_vehicles(self._vehicle, search_distance)
        oncoming_list = [v for v in vehicle_list if
                         self._map.get_waypoint(v.get_location()).lane_id == ego_wp.lane_id * -1]

        if not oncoming_list: return None

        ego_front_transform = _extend_bounding_box(self._vehicle)
        for vehicle in oncoming_list:
            target_front_transform = _extend_bounding_box(vehicle)
            if is_within_distance(target_front_transform, ego_front_transform, search_distance, angle_interval=[0, 90]):
                return vehicle
        return None

    @staticmethod
    def _calculate_spatial_clearance(v_length, l_width, base_offset, opposite_offset, proximity_margin):
        hypotenuse = math.sqrt(v_length ** 2 + l_width ** 2)
        total_clearance = proximity_margin + base_offset + hypotenuse + opposite_offset + hypotenuse
        return total_clearance, hypotenuse

    @staticmethod
    def _estimate_maneuver_duration(ego_vehicle, total_clearance):
        v0 = get_speed(ego_vehicle) / 3.6
        a = 3.5
        return (-v0 + math.sqrt(v0 ** 2 + 2 * a * total_clearance)) / a