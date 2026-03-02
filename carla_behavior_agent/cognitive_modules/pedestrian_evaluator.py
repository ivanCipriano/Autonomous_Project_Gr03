# cognitive_modules/pedestrian_evaluator.py
from .base_evaluator import BaseEvaluator
from misc import compute_distance_from_center, is_within_distance, dist
from local_planner import RoadOption


class BipedalHazardEvaluator(BaseEvaluator):
    """Sostituisce lo Scenario 2: Rilevamento e mitigazione pedoni"""

    def _scan_for_bipeds(self, current_waypoint):
        """Ex pedestrian_avoid_manager: Scansiona l'ambiente alla ricerca di pedoni"""
        sys = self.core_system

        # Recupero entità bipedali
        biped_list = sys._world.get_actors().filter("*walker.pedestrian*")

        # Calcolo del raggio di ricerca dinamico basato sul contesto (incrocio o navigazione libera)
        scan_radius = 12 if sys._navigation_engine.is_traversing else 20

        # Filtro spaziale
        biped_list = [w for w in biped_list if
                      is_within_distance(w.get_transform(), sys._vehicle.get_transform(), scan_radius,
                                         angle_interval=[0, 90])]

        if not biped_list:
            return False, None, -1

        v_speed_limit = sys._vehicle.get_speed_limit()

        if current_waypoint.is_junction:
            return True, biped_list[0], dist(biped_list[0], current_waypoint)

        elif sys._direction == RoadOption.CHANGELANELEFT:
            return sys._vehicle_obstacle_detected(biped_list,
                                                  max(sys._behavior.min_proximity_threshold, v_speed_limit / 2),
                                                  up_angle_th=90, lane_offset=-1)

        elif sys._direction == RoadOption.CHANGELANERIGHT:
            return sys._vehicle_obstacle_detected(biped_list,
                                                  max(sys._behavior.min_proximity_threshold, v_speed_limit / 2),
                                                  up_angle_th=90, lane_offset=1)

        else:
            return sys._vehicle_obstacle_detected(biped_list,
                                                  max(sys._behavior.min_proximity_threshold, v_speed_limit / 2),
                                                  up_angle_th=90)

    def evaluate(self, **kwargs):
        current_waypoint = kwargs.get('ego_vehicle_wp')
        if not current_waypoint:
            return None

        # Rilevamento entità bipedali usando il metodo interno appena integrato
        hazard_detected, entity, raw_dist = self._scan_for_bipeds(current_waypoint)

        if hazard_detected:
            actual_distance = compute_distance_from_center(
                actor1=self.core_system._vehicle,
                actor2=entity,
                distance=raw_dist
            )

            print(f"[Cognition] -> Bipedal entity track [{entity.id}] detected at {actual_distance:.2f} units.")

            if actual_distance < self.core_system._behavior.min_proximity_threshold:
                print("--- [Cognition] Collision trajectory imminent. Initiating max-brake protocol.")
                return self.halt_vehicle()
            else:
                print("--- [Cognition] Entering caution zone. Decelerating to approach velocity.")
                self.core_system.set_target_speed(self.core_system._approach_speed)
                return self.core_system._local_planner.run_step()

        return None