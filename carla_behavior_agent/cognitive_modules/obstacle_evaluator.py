from .base_evaluator import BaseEvaluator
from misc import compute_distance_from_center, is_within_distance, get_distance


class StaticObstacleEvaluator(BaseEvaluator):
    """
    Classe responsabile della valutazione e gestione degli ostacoli statici presenti nell'ambiente di simulazione.

    Eredita da BaseEvaluator e costituisce un componente fondamentale del modulo di percezione e decisione del veicolo
    autonomo. Il suo scopo principale è rilevare elementi stazionari di pericolo, come coni stradali o segnali di
    avviso (traffic warnings), e calcolare le risposte comportamentali appropriate, quali l'evitamento dell'ostacolo
    tramite ricalcolo della traiettoria globale (bypass) o l'applicazione di un offset laterale locale per la sicurezza.
    Questa classe implementa la logica necessaria per risolvere scenari critici legati all'ostruzione della carreggiata.
    """

    def _detect_static_obstacles(self, current_waypoint, element_query, angle_interval=[0, 90]):
        """
        Rileva la presenza di pericoli statici specifici nell'area circostante l'ego-veicolo.

        Il metodo interroga l'ambiente simulato (CARLA world) filtrando gli attori in base alla stringa di ricerca fornita.
        Successivamente, isola gli elementi che si trovano entro un raggio di 20 metri e all'interno di un settore angolare
        specifico rispetto alla posizione e all'orientamento dell'ego-veicolo. Gli ostacoli rilevati vengono ordinati per
        distanza dal waypoint corrente. Presenta una logica di elaborazione differenziata per i coni stradali rispetto
        ad altri elementi di intralcio generici.

        Parametri:
            current_waypoint (carla.Waypoint): Il waypoint in cui si trova attualmente l'ego-veicolo.
            element_query (str): Stringa utilizzata per filtrare gli attori di CARLA (es. "static.prop.constructioncone").
            angle_interval (list, opzionale): Intervallo angolare [min, max] in gradi per delimitare il campo visivo
            della ricerca rispetto all'orientamento del veicolo. Il valore predefinito è [0, 90].

        Ritorna:
            tuple: Una tupla contenente tre elementi:
            - bool: True se un ostacolo rilevante è stato individuato, False altrimenti.
            - carla.Actor o None: L'oggetto CARLA rappresentante l'ostacolo più vicino, se presente.
            - float: La distanza in metri dall'ostacolo (oppure -1 se non ci sono ostacoli validi).
        """
        sys = self.core_system
        obs_list = sys._world.get_actors().filter(element_query)
        obs_list = [o for o in obs_list if is_within_distance(o.get_transform(), sys._vehicle.get_transform(),
                    sys._obstacle_scan_radius, angle_interval=angle_interval)]

        if not obs_list:
            return False, None, -1

        obs_list = sorted(obs_list, key=lambda x: get_distance(x, current_waypoint))

        if "constructioncone" in element_query:
            return True, obs_list[0], get_distance(obs_list[0], current_waypoint)

        return sys._vehicle_obstacle_detected(obs_list, max(sys._behavior.min_proximity_threshold, sys._speed_limit / 2))

    def evaluate(self, **kwargs):
        """
        Analizza l'ambiente circostante per identificare ostacoli statici ed esegue le manovre evasive necessarie.

        Il metodo coordina la scansione dell'ambiente alla ricerca di cartelli di avviso e coni da cantiere. I risultati
        della scansione vengono memorizzati nel sistema centrale per essere condivisi con altri valutatori della pipeline.
        In caso di rilevamento di un cartello di avviso sulla traiettoria, calcola un percorso di evitamento (bypass).
        Se l'evitamento non è fisicamente fattibile e l'ostacolo è in zona critica (< 6 metri), ordina un arresto
        di emergenza. In presenza di coni da cantiere, applica dinamicamente uno scostamento laterale (offset) al planner
        locale per aggirare i detriti, modulando la direzione in base all'appartenenza del cono alla corsia di marcia.

        Parametri:
            **kwargs: Dizionario di argomenti chiave variabili. I parametri attesi includono:
            - ego_vehicle_wp (carla.Waypoint): Il waypoint corrente in cui si trova l'ego-veicolo.

        Ritorna:
            Un'azione esecutiva per il controller del veicolo (es. il richiamo a halt_vehicle() per l'arresto immediato)
            in caso di situazioni ad alto rischio, oppure None se l'area è sicura o se l'ostacolo viene interamente gestito
            tramite l'applicazione di un offset laterale delegato al planner locale.
        """
        current_waypoint = kwargs.get('ego_vehicle_wp')
        if not current_waypoint:
            return None

        sys = self.core_system
        hazard_flag, hazard_obj, hazard_dist = self._detect_static_obstacles(current_waypoint,
                                                                           "*static.prop.trafficwarning*")
        cone_flag, cone_obj, _ = self._detect_static_obstacles(current_waypoint, "*static.prop.constructioncone*")

        sys.environmental_hazards = {'tw_state': hazard_flag, 'cone_state': cone_flag}

        if hazard_flag is True:
            actual_dist = compute_distance_from_center(actor1=sys._vehicle, actor2=hazard_obj, distance=hazard_dist)

            evasion_path = sys._overtaking_engine.compute_evasion_trajectory(hazard_obj, current_waypoint, 1, 20, actual_dist, sys._speed_limit)

            if evasion_path:
                print("[Cognition] -> Trajectory alteration: bypassing environmental hazard.")
                sys._BehaviorAgent__update_global_plan(overtake_path=evasion_path)

            if not sys._overtaking_engine.is_overtaking and actual_dist < sys._obstacle_critical_dist:
                print("--- [Cognition] Bypass unfeasible. Executing critical halt.")
                return self.halt_vehicle()

        elif not hazard_flag and cone_flag and not sys._overtaking_engine.is_overtaking:
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