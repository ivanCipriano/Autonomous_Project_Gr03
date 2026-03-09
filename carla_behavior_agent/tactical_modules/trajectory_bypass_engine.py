import carla
import math
from basic_agent import BasicAgent
from misc import get_distance, is_within_distance, compute_distance_from_center, get_speed


class TrajectoryBypassEngine(BasicAgent):
    """
    Motore tattico dedicato alla pianificazione e generazione di traiettorie per l'elusione e il sorpasso di ostacoli.
    Gestisce in autonomia la stima degli spazi, il controllo del traffico in senso opposto e la creazione del percorso.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Inizializza il motore di elusione e sorpasso.

        Args:
            vehicle (carla.Vehicle): L'attore veicolo controllato dall'agente.
            opt_dict (dict, opzionale): Dizionario contenente opzioni di configurazione aggiuntive. Default a {}.
            map_inst (carla.Map, opzionale): Istanza della mappa CARLA per interrogazioni spaziali. Default a None.
            grp_inst (GlobalRoutePlanner, opzionale): Istanza del pianificatore di percorso globale. Default a None.

        Returns:
            None
        """
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._evasion_lock_frames = 0
        self._is_executing_bypass = False
        self._required_clearance = 0
        self._bypass_safety_margin = opt_dict.get('bypass_safety_margin', 3.0)
        self._bypass_search_radius = opt_dict.get('bypass_search_radius', 30.0)
        self._ego_acceleration_estimate = opt_dict.get('ego_acceleration_estimate', 3.5)
        self._base_sign_threshold = opt_dict.get('base_sign_threshold', 10.0)

    def compute_evasion_trajectory(self, target_entity: carla.Actor, current_wp: carla.Waypoint,
                                   base_offset: float = 1, opposite_offset: float = 0,
                                   proximity_margin: float = 18, max_velocity: float = 50):
        """
        Genera una traiettoria sicura per sorpassare o aggirare un ostacolo.
        Calcola gli spazi necessari, stima i tempi della manovra, verifica il traffico in arrivo
        nella corsia opposta e si assicura che non vi siano incroci critici nel percorso.

        Args:
            target_entity (carla.Actor): L'ostacolo o il veicolo lento da sorpassare.
            current_wp (carla.Waypoint): Il waypoint attuale del veicolo controllato (ego vehicle).
            base_offset (float, opzionale): Distanza di offset iniziale per il cambio di corsia. Default a 1.
            opposite_offset (float, opzionale): Distanza di avanzamento necessaria nella corsia opposta. Se 0, viene calcolata dinamicamente. Default a 0.
            proximity_margin (float, opzionale): Margine di sicurezza frontale prima di iniziare la manovra. Default a 18.
            max_velocity (float, opzionale): Velocità massima stimata (in km/h) dei veicoli in senso opposto. Default a 50.

        Returns:
            list o None: Lista di carla.Waypoint che definiscono la traiettoria di elusione, oppure None se la manovra non è sicura.
        """

        try:
            is_target_at_stop, _ = self._affected_by_sign(vehicle=target_entity, sign_type="206", max_distance=self._base_sign_threshold)
            if is_target_at_stop:
                print(f"[MANOVRA] Sorpasso annullato: il veicolo target (ID: {target_entity.id}) si trova ad uno STOP.")
                return None
        except AttributeError:
            pass
        
        if not opposite_offset:
            opposite_offset = self._estimate_opposite_clearance(target_entity, self._bypass_search_radius)

        v_length = self._vehicle.bounding_box.extent.x
        l_width = current_wp.lane_width

        self._required_clearance, hypotenuse = self._calculate_spatial_clearance(v_length, l_width, base_offset, opposite_offset, proximity_margin)
        maneuver_time = self._estimate_maneuver_duration(self._vehicle, self._required_clearance, self._ego_acceleration_estimate)

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
                                                        reference_transform=next_ego_wp.transform,
                                                        max_distance=self._bypass_search_radius,
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
            
            target_id = target_entity.id
            target_type = target_entity.type_id if hasattr(target_entity, 'type_id') else "Sconosciuto"
            print(f"\n[MANOVRA] Inizio manovra di sorpasso!")
            print(f"  -> Veicolo target - ID: {target_id}, Tipo: {target_type}")
            print(f"  -> Spazio richiesto: {self._required_clearance:.2f}m")
            print(f"  -> Tempo stimato: {maneuver_time:.2f}s")

            return bypass_path

    @property
    def is_bypassing(self):
        """
        Proprietà che indica se il veicolo sta attualmente eseguendo una manovra di sorpasso/elusione.

        Returns:
            bool: True se la manovra è in corso, False altrimenti.
        """
        return self._is_executing_bypass

    @is_bypassing.setter
    def is_bypassing(self, val):
        """
        Imposta lo stato di esecuzione della manovra di sorpasso.

        Args:
            val (bool): Nuovo stato della manovra.

        Returns:
            None
        """
        self._is_executing_bypass = val

    @property
    def evasion_lock(self):
        """
        Proprietà che restituisce il numero di frame per i quali la logica di elusione è "bloccata" (ossia la manovra deve completarsi).

        Returns:
            int: Numero di frame di blocco rimanenti.
        """
        return self._evasion_lock_frames

    @evasion_lock.setter
    def evasion_lock(self, val):
        """
        Imposta il numero di frame di blocco per la manovra elusiva.

        Args:
            val (int): Numero di frame desiderati.

        Returns:
            None
        """
        self._evasion_lock_frames = val

    @property
    def required_clearance(self):
        """
        Proprietà che restituisce lo spazio libero (clearance) richiesto per completare il sorpasso in sicurezza.

        Returns:
            float: Distanza necessaria in metri.
        """
        return self._required_clearance

    def _estimate_opposite_clearance(self, actor, max_distance = 30):
        """
        Stima la lunghezza del tratto nella corsia opposta necessario per superare l'ostacolo.
        Considera la lunghezza dell'attore target e di eventuali altri veicoli parcheggiati adiacenti ad esso.

        Args:
            actor (carla.Actor): L'ostacolo primario da cui calcolare la distanza.
            max_distance (float, opzionale): Distanza massima entro cui cercare altri veicoli adiacenti. Default a 30.

        Returns:
            float: Distanza totale stimata necessaria nella corsia opposta (include un margine di sicurezza di 3 metri).
        """
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
                v_distance = compute_distance_from_center(actor1=previous_vehicle, actor2=v, distance=get_distance(v, previous_vehicle))
            else:
                continue
            distance_other_lane += v.bounding_box.extent.x + v_distance
            previous_vehicle = v

        return distance_other_lane + self._bypass_safety_margin

    def _detect_oncoming_traffic(self, ego_wp, search_distance = 30):
        """
        Analizza la corsia opposta per individuare veicoli in avvicinamento (oncoming traffic).
        Utilizza un'estensione virtuale dei bounding box lungo i vettori forward per prevedere collisioni imminenti.

        Args:
            ego_wp (carla.Waypoint): Il waypoint corrente del veicolo ego.
            search_distance (float, opzionale): Raggio di ricerca frontale in metri. Default a 30.

        Returns:
            carla.Actor o None: L'attore veicolo rilevato in senso opposto, oppure None se la corsia è libera.
        """
        def _extend_bounding_box(actor):
            wp = self._map.get_waypoint(actor.get_location())
            transform = wp.transform
            forward_vector = transform.get_forward_vector()
            extent = actor.bounding_box.extent.x
            transform.location += carla.Location(x=extent * forward_vector.x, y=extent * forward_vector.y)
            return transform

        vehicle_list = self._get_ordered_vehicles(self._vehicle, search_distance)
        oncoming_list = [v for v in vehicle_list if self._map.get_waypoint(v.get_location()).lane_id == ego_wp.lane_id * -1]

        if not oncoming_list: return None

        ego_front_transform = _extend_bounding_box(self._vehicle)
        for vehicle in oncoming_list:
            target_front_transform = _extend_bounding_box(vehicle)
            if is_within_distance(target_front_transform, ego_front_transform, search_distance, angle_interval=[0, 90]):
                return vehicle
        return None

    @staticmethod
    def _calculate_spatial_clearance(v_length, l_width, base_offset, opposite_offset, proximity_margin):
        """
        Calcola la distanza spaziale totale richiesta per l'intera manovra geometrica, includendo
        l'ingresso, la percorrenza nella corsia opposta e il rientro, sfruttando l'ipotenusa del triangolo di cambio corsia.

        Args:
            v_length (float): Lunghezza dell'ego vehicle.
            l_width (float): Larghezza della corsia.
            base_offset (float): Offset longitudinale per il primo cambio corsia.
            opposite_offset (float): Distanza lineare da percorrere nella corsia di sorpasso.
            proximity_margin (float): Margine di sicurezza iniziale.

        Returns:
            tuple: Una tupla (total_clearance, hypotenuse) dove total_clearance è la distanza totale richiesta e hypotenuse è la distanza diagonale per il cambio corsia.
        """
        hypotenuse = math.sqrt(v_length ** 2 + l_width ** 2)
        total_clearance = proximity_margin + base_offset + hypotenuse + opposite_offset + hypotenuse
        return total_clearance, hypotenuse

    @staticmethod
    def _estimate_maneuver_duration(ego_vehicle, total_clearance, a=3.5):
        """
        Stima il tempo necessario (in secondi) per completare l'intera manovra di sorpasso,
        utilizzando le equazioni del moto uniformemente accelerato.


        Args:
            ego_vehicle (carla.Vehicle): L'attore del veicolo controllato (per leggerne la velocità corrente).
            total_clearance (float): La distanza totale (in metri) calcolata per la manovra.

        Returns:
            float: Tempo stimato in secondi.
        """
        v0 = get_speed(ego_vehicle) / 3.6
        return (-v0 + math.sqrt(v0 ** 2 + 2 * a * total_clearance)) / a