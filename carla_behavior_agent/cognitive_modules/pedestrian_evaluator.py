from .base_evaluator import BaseEvaluator
from misc import compute_distance_from_center, is_within_distance, get_distance
from local_planner import RoadOption


class BipedalHazardEvaluator(BaseEvaluator):
    """
    Valuta e gestisce i pericoli associati alla presenza di pedoni nell'ambiente.

    Questa classe estende BaseEvaluator per implementare una logica specifica di
    scansione e reazione ai pedoni, calcolando la distanza e determinando se il veicolo
    deve frenare bruscamente o semplicemente rallentare.
    """

    def _scan_for_bipeds(self, current_waypoint):
        """
        Scansiona l'ambiente circostante per rilevare la presenza di pedoni nelle vicinanze.

        Il raggio di scansione varia dinamicamente a seconda se il veicolo
        sta attraversando un incrocio o meno. Considera inoltre l'intenzione di sterzata
        (es. cambio corsia) per calcolare correttamente gli ostacoli sulla traiettoria.

        Args:
            current_waypoint (Waypoint): Il waypoint corrente su cui si trova l'ego-vehicle.

        Returns:
            tuple: Una tupla contenente tre elementi:
                - bool: True se è stato rilevato un pedone pericoloso, False altrimenti.
                - object o None: L'oggetto che rappresenta il pedone rilevato, None se assente.
                - float: La distanza dal pedone rilevato, oppure -1 se non ci sono pedoni.
        """
        sys = self.core_system

        biped_list = sys._world.get_actors().filter("*walker.pedestrian*")
        scan_radius = sys._pedestrian_scan_junction if sys._navigation_engine.is_traversing else sys._pedestrian_scan_normal
        biped_list = [w for w in biped_list if
                      is_within_distance(w.get_transform(), sys._vehicle.get_transform(), scan_radius,
                                         angle_interval=[0, 90])]

        if not biped_list:
            return False, None, -1

        v_speed_limit = sys._vehicle.get_speed_limit()

        if current_waypoint.is_junction:
            return True, biped_list[0], get_distance(biped_list[0], current_waypoint)

        elif sys._direction == RoadOption.CHANGELANELEFT:
            return sys._vehicle_obstacle_detected(biped_list, max(sys._behavior.min_proximity_threshold, v_speed_limit / 2),
                                                  up_angle_th=90, lane_offset=-1)

        elif sys._direction == RoadOption.CHANGELANERIGHT:
            return sys._vehicle_obstacle_detected(biped_list, max(sys._behavior.min_proximity_threshold, v_speed_limit / 2),
                                                  up_angle_th=90, lane_offset=1)

        else:
            return sys._vehicle_obstacle_detected(biped_list, max(sys._behavior.min_proximity_threshold, v_speed_limit / 2),
                                                  up_angle_th=90)

    def evaluate(self, **kwargs):
        """
        Valuta la situazione corrente rispetto ai pedoni circostanti e determina l'azione
        di guida appropriata.

        Se viene rilevato un pedone, calcola la distanza reale dai centri fisici.
        Se il pedone è sotto la soglia minima di prossimità, attiva il protocollo di
        frenata d'emergenza. Altrimenti, rallenta il veicolo alla velocità di approccio.

        Args:
            **kwargs: Argomenti chiave. Deve contenere 'ego_vehicle_wp' (il waypoint
                      corrente del veicolo).

        Returns:
            object o None: Restituisce un comando di frenata, il passo di esecuzione del local planner locale,
                            oppure None se non viene rilevato alcun pericolo.
        """
        current_waypoint = kwargs.get('ego_vehicle_wp')
        if not current_waypoint:
            return None

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