# tactical_modules/intersection_navigation_engine.py
import carla
import math
import numpy as np
from enum import IntEnum
from typing import List

from basic_agent import BasicAgent
from misc import get_distance, get_speed
from local_planner import RoadOption


class IntersectionTopology(IntEnum):
    NULL_STATE = -1
    AHEAD_PORT = 0
    AHEAD_STARBOARD = 1
    PORT_STARBOARD = 2
    OMNIDIRECTIONAL = 3


class IntersectionNavigationEngine(BasicAgent):
    """
    Motore tattico per la negoziazione e l'attraversamento degli incroci.
    (Sostituisce ex JunctionManager)
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._active_traversal_frames = 0
        self._is_traversing = False
        self._target_node = None
        self._topology_id = -1
        self._halt_frames = 0
        self._mandatory_wait_time = 2
        self._starboard_turn_frames = 0

    def evaluate_traversal_safety(self, local_planner, preview_steps: int) -> bool:
        """Ex run_step: Valuta se è sicuro impegnare l'incrocio"""
        self._local_planner = local_planner
        _, routing_directive = self._local_planner.get_incoming_waypoint_and_direction(steps=preview_steps)

        # Sosta obbligatoria pre-attraversamento
        if self._halt_frames < (self._mandatory_wait_time / self._world.get_snapshot().timestamp.delta_seconds):
            self._halt_frames += 1
            return False

        if not self._is_traversing:
            ego_wp = self._map.get_waypoint(self._vehicle.get_location())
            idx = 1
            while not ego_wp.next(idx)[0].is_junction: idx += 1
            self._target_node = ego_wp.next(idx)[0]

        intersection = self._target_node.get_junction()
        if not intersection:
            return True

        center_transform = carla.Transform(intersection.bounding_box.location, carla.Rotation())
        radius = math.sqrt(intersection.bounding_box.extent.y ** 2 + intersection.bounding_box.extent.x ** 2)

        occupying_fleet = self._get_ordered_vehicles(center_transform, radius)
        occupying_fleet = [v for v in occupying_fleet if
                           self._map.get_waypoint(v.get_location()).junction_id == intersection.id]

        self._topology_id = self._classify_topology(self._target_node, intersection)
        entry_nodes = self._map_entry_nodes(intersection)

        pivot = carla.Transform(intersection.bounding_box.location, self._target_node.transform.rotation)
        spatial_nodes = self._categorize_spatial_nodes(pivot, list(entry_nodes))

        if self._topology_id == IntersectionTopology.NULL_STATE:
            return True

        # ==========================================
        # Valutazione Topologica dell'Incrocio
        # ==========================================
        ahead_node = spatial_nodes['front'][0] if spatial_nodes['front'] else None
        port_node = spatial_nodes['left'][0] if spatial_nodes['left'] else None
        starboard_node = spatial_nodes['right'][0] if spatial_nodes['right'] else None

        v_ahead = self._get_ordered_vehicles(ahead_node, 9)[0] if ahead_node and self._get_ordered_vehicles(ahead_node,
                                                                                                            9) else None
        v_port = self._get_ordered_vehicles(port_node, 9)[0] if port_node and self._get_ordered_vehicles(port_node,
                                                                                                         9) else None
        v_starboard = self._get_ordered_vehicles(starboard_node, 9)[0] if starboard_node and self._get_ordered_vehicles(
            starboard_node, 9) else None

        if self._topology_id == IntersectionTopology.AHEAD_PORT:
            if not v_ahead and not v_port: return True
            if routing_directive == RoadOption.LEFT:
                if not self._check_linear_trajectory(v_ahead) and not self._check_starboard_signal(
                    v_ahead) and not self._check_port_signal(v_port): return True
            elif routing_directive == RoadOption.STRAIGHT:
                if not self._check_port_signal(v_port): return True
            return False

        elif self._topology_id == IntersectionTopology.AHEAD_STARBOARD:
            if not v_ahead and not v_starboard: return True
            if routing_directive == RoadOption.RIGHT:
                if not self._check_port_signal(v_ahead) or (
                        self._check_port_signal(v_starboard) and get_speed(v_starboard) > 0.1): return True
            elif routing_directive == RoadOption.STRAIGHT:
                if not self._check_port_signal(v_starboard) and not self._check_starboard_signal(
                    v_starboard) and not self._check_port_signal(v_ahead): return True
            return False

        elif self._topology_id == IntersectionTopology.PORT_STARBOARD:
            if not v_port and not v_starboard: return True
            if routing_directive == RoadOption.LEFT:
                if not self._check_linear_trajectory(v_port) and not self._check_linear_trajectory(
                    v_starboard) and not self._check_port_signal(v_starboard):
                    return True
                elif self._check_starboard_signal(v_port) and v_starboard in occupying_fleet:
                    return True
            elif routing_directive == RoadOption.RIGHT:
                if not self._check_linear_trajectory(v_port): return True
            return False

        elif self._topology_id == IntersectionTopology.OMNIDIRECTIONAL:
            if not v_ahead and not v_port and not v_starboard: return True
            if routing_directive == RoadOption.STRAIGHT:
                if not self._check_port_signal(v_port) and not self._check_starboard_signal(
                        v_starboard) and not self._check_port_signal(v_ahead) \
                        and not self._check_linear_trajectory(v_port) and not self._check_linear_trajectory(
                    v_starboard):
                    return True
                elif self._check_starboard_signal(v_port) and v_port in occupying_fleet:
                    return True
                elif (self._check_linear_trajectory(v_ahead) or self._check_starboard_signal(
                    v_ahead)) and v_ahead in occupying_fleet:
                    return True
            elif routing_directive == RoadOption.LEFT:
                if not self._check_linear_trajectory(v_ahead) and not self._check_linear_trajectory(
                        v_port) and not self._check_linear_trajectory(v_starboard) \
                        and not self._check_starboard_signal(v_ahead) and not self._check_port_signal(
                    v_port) and not self._check_port_signal(v_starboard):
                    return True
                elif (self._check_port_signal(v_ahead) or self._check_starboard_signal(
                    v_ahead)) and v_ahead in occupying_fleet:
                    return True
                elif self._check_starboard_signal(v_port) and v_port in occupying_fleet:
                    return True
                elif self._check_starboard_signal(v_starboard) and v_starboard in occupying_fleet:
                    return True
            elif routing_directive == RoadOption.RIGHT:
                if not self._check_linear_trajectory(v_port) and not self._check_port_signal(v_ahead): return True
            return False

        return False

    # Getters / Setters Offuscati
    @property
    def is_traversing(self):
        return self._is_traversing

    @is_traversing.setter
    def is_traversing(self, val):
        self._is_traversing = val

    @property
    def active_traversal_frames(self):
        return self._active_traversal_frames

    @active_traversal_frames.setter
    def active_traversal_frames(self, val):
        self._active_traversal_frames = val

    @property
    def halt_frames(self):
        return self._halt_frames

    @halt_frames.setter
    def halt_frames(self, val):
        self._halt_frames = val

    @property
    def topology_id(self):
        return self._topology_id

    # --- Metodi Privati Rinominati ---
    def _map_entry_nodes(self, intersection):
        junction_wps = intersection.get_waypoints(carla.LaneType.Driving)
        entry_wps = set()
        seen_coords = set()
        for begin, _ in junction_wps:
            key = (round(begin.transform.location.x, 2), round(begin.transform.location.y, 2),
                   round(begin.transform.location.z, 2))
            if key in seen_coords: continue
            entry_wps.add(begin)
            seen_coords.add(key)
        return entry_wps

    def _categorize_spatial_nodes(self, point, neighbors, direction=None):
        dir_neighbors = {'left': [], 'right': [], 'front': [], 'back': []}
        pivot_pos = np.array([point.location.x, point.location.y, point.location.z])
        pivot_yaw = math.radians(point.rotation.yaw)

        for n in neighbors:
            v = np.array([n.transform.location.x, n.transform.location.y, n.transform.location.z]) - pivot_pos
            u = np.array([np.cos(pivot_yaw), np.sin(pivot_yaw), 0])
            cross = np.cross(u, v)[2]
            inner = np.inner(u, v)
            if abs(cross) < abs(inner):
                dir_neighbors['front' if inner > 0 else 'back'].append(n)
            else:
                dir_neighbors['left' if cross < 0 else 'right'].append(n)
        return dir_neighbors if direction is None else dir_neighbors[direction]

    def _classify_topology(self, vehicle_wp, intersection):
        wps_in_junction = intersection.get_waypoints(carla.LaneType.Driving)
        outgoing_wps = []
        outgoing_wps.extend([wp2 for wp1, wp2 in wps_in_junction if get_distance(wp1, vehicle_wp) < 3])
        outgoing_wps.extend([wp1 for wp1, wp2 in wps_in_junction if get_distance(wp2, vehicle_wp) < 3])
        pivot = carla.Transform(intersection.bounding_box.location, vehicle_wp.transform.rotation)
        oriented = self._categorize_spatial_nodes(pivot, outgoing_wps)

        if oriented['left'] and not oriented['right'] and oriented['front']:
            return IntersectionTopology.AHEAD_PORT
        elif not oriented['left'] and oriented['right'] and oriented['front']:
            return IntersectionTopology.AHEAD_STARBOARD
        elif oriented['left'] and oriented['right'] and not oriented['front']:
            return IntersectionTopology.PORT_STARBOARD
        elif oriented['left'] and oriented['right'] and oriented['front']:
            return IntersectionTopology.OMNIDIRECTIONAL
        return IntersectionTopology.NULL_STATE

    # --- Metodi Percettivi Statici Rinominati ---
    @staticmethod
    def _check_linear_trajectory(vehicle):
        if not vehicle: return False
        light_state = vehicle.get_light_state()
        if bool(light_state & carla.libcarla.VehicleLightState.LeftBlinker) or bool(
            light_state & carla.libcarla.VehicleLightState.RightBlinker): return False
        return True

    @staticmethod
    def _check_port_signal(vehicle):
        if not vehicle: return False
        return bool(vehicle.get_light_state() & carla.libcarla.VehicleLightState.LeftBlinker)

    @staticmethod
    def _check_starboard_signal(vehicle):
        if not vehicle: return False
        return bool(vehicle.get_light_state() & carla.libcarla.VehicleLightState.RightBlinker)

    @property
    def starboard_turn_frames(self):
        return self._starboard_turn_frames

    @starboard_turn_frames.setter
    def starboard_turn_frames(self, val):
        self._starboard_turn_frames = val

    def execute_traversal(self, routing_directive, lock_frames: float = 120) -> carla.VehicleControl:
        """Inizia l'impegno dell'incrocio (ex __initiate_intersection_traversal)"""
        self._active_traversal_frames = lock_frames
        self._is_traversing = True

        if routing_directive == RoadOption.RIGHT and self._topology_id == IntersectionTopology.OMNIDIRECTIONAL:
            self._starboard_turn_frames = round(2 / self._world.get_snapshot().timestamp.delta_seconds)
            self._local_planner.set_lateral_offset(-0.3)

        return self._local_planner.run_step()