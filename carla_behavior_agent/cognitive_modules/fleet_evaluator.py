# cognitive_modules/fleet_evaluator.py
from .base_evaluator import BaseEvaluator
from misc import compute_distance_from_center, get_speed


class FleetProximityEvaluator(BaseEvaluator):
    """
    Classe responsabile della valutazione di prossimità e delle interazioni con altri veicoli o elementi mobili nell'ambiente.

    Eredita da BaseEvaluator e svolge un ruolo cruciale nel modulo di percezione e decisione del sistema di guida autonoma.
    Il suo scopo principale è analizzare le entità circostanti per garantire la sicurezza della navigazione. Gestisce in
    modo dinamico scenari di traffico complessi, valutando le distanze di sicurezza relative, pianificando manovre di
    evitamento ostacoli (bypass), sorpassi di veicoli parcheggiati o biciclette, e regolando l'attivazione del sistema
    di cruise control adattivo in risposta al comportamento del traffico locale.
    """
    def evaluate(self, **kwargs):
        """
        Analizza l'ambiente circostante per rilevare ostacoli dinamici e determinare la reazione ottimale dell'ego-veicolo.

        Il metodo esegue una scansione della flotta di veicoli nelle vicinanze. Se un target viene individuato, ne calcola
        la distanza e la posizione relativa, innescando una logica decisionale contestuale:
        - Ostacoli vulnerabili (Biciclette): valuta se occupano il centro della corsia per calcolare una traiettoria
        di bypass evasiva completa, o se sono posizionati a lato, calcolando un semplice scostamento laterale.
        - Ostacoli statici (Veicoli parcheggiati): genera traiettorie di sorpasso se la corsia lo consente.
        - Situazioni di emergenza: impone un arresto immediato (halt_vehicle) se la distanza scende sotto la soglia critica.
        - Traffico regolare: attiva il sistema di Adaptive Cruise Control (ACC) per il mantenimento della distanza di sicurezza.

        Parametri:
            **kwargs: Dizionario di argomenti chiave variabili. I parametri attesi includono:
                - ego_vehicle_wp (carla.Waypoint): Il waypoint corrente in cui si trova l'ego-veicolo.
                - debug (bool, opzionale): Flag per l'abilitazione dei log e delle informazioni visive (default: False).

        Ritorna:
            Un'azione esecutiva per il controller del veicolo, che può tradursi nel passaggio al planner locale, in una
            manovra di arresto, o nel mantenimento di un comportamento di guida normale. Ritorna None se l'area è libera.
        """
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
        Verifica la linearità del tratto stradale attuale tramite il confronto degli angoli di imbardata dei veicoli coinvolti.

        Nel contesto della guida autonoma, questo controllo geometrico è fondamentale per le valutazioni di sorpasso.
        Assumendo che i veicoli procedano allineati alla strada, una minima deviazione tra le rispettive rotazioni
        sull'asse Z indica che la via è rettilinea, condizione necessaria per avviare manovre di bypass in sicurezza
        evitando deviazioni improvvise dovute alla curvatura della corsia.

        Parametri:
            ego_yaw (float): Angolo di imbardata (yaw) dell'ego-veicolo, misurato in gradi.
            vehicle_yaw (float): Angolo di imbardata (yaw) del veicolo target (es. bicicletta), misurato in gradi.
            tolerance (int, opzionale): Soglia massima di differenza angolare consentita in gradi. Default a 10.

        Ritorna:
            bool: True se la differenza assoluta tra gli angoli è strettamente inferiore alla tolleranza, False altrimenti.
        """
        return abs(ego_yaw - vehicle_yaw) < tolerance


    def is_bicycle_near_center(self,vehicle_location, ego_vehicle_wp):
        """
        Determina se una bicicletta si trova nella porzione centrale della corsia percorsa dall'ego-veicolo.

        Il metodo calcola lo scostamento trasversale (sull'asse Y) tra le coordinate fisiche del mezzo vulnerabile rilevato
        e il centro ideale della corsia di marcia (rappresentato dal waypoint dell'ego-veicolo). Questa informazione guida
        il sistema decisionale: una bicicletta centrale richiede una vera e propria invasione della corsia adiacente
        (bypass trajectory), mentre una bicicletta sul margine stradale può essere superata con un lieve scostamento interno.

        Parametri:
            vehicle_location (carla.Location): L'oggetto contenente le coordinate spaziali attuali del veicolo target.
            ego_vehicle_wp (carla.Waypoint): Il waypoint di riferimento dell'ego-veicolo.

        Ritorna:
            bool: True se la distanza trasversale sull'asse Y risulta inferiore a una tolleranza prefissata (es. 0.3 metri),
                False se il veicolo target è posizionato più vicini ai margini laterali della carreggiata.
        """
        lane_center_offset = 0.3  # How close to the center the bicycle needs to be considered in the center
        vehicle_y = vehicle_location.y
        lane_center_y = ego_vehicle_wp.transform.location.y
        return abs(vehicle_y - lane_center_y) < lane_center_offset