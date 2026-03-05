import carla
import math
import numpy as np
from enum import IntEnum
from basic_agent import BasicAgent
from misc import get_distance, get_speed
from local_planner import RoadOption


class IntersectionTopology(IntEnum):
    """
    Enumerazione che definisce la topologia di un incrocio in base alle direzioni di uscita disponibili.
    """
    NULL_STATE = -1
    AHEAD_PORT = 0
    AHEAD_STARBOARD = 1
    PORT_STARBOARD = 2
    OMNIDIRECTIONAL = 3


class IntersectionNavigationEngine(BasicAgent):
    """
    Motore tattico per la gestione, negoziazione e attraversamento sicuro degli incroci.
    Gestisce il comportamento del veicolo in base alla presenza di altri veicoli e alla geometria dell'incrocio.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Inizializza il motore di navigazione per gli incroci.

        Args:
            vehicle (carla.Vehicle): L'attore veicolo controllato dall'agente.
            opt_dict (dict, opzionale): Dizionario di opzioni per la configurazione dell'agente. Default è {}.
            map_inst (carla.Map, opzionale): Istanza della mappa di CARLA. Default è None.
            grp_inst (GlobalRoutePlanner, opzionale): Istanza del pianificatore di percorso globale. Default è None.

        Returns:
            None
        """
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._active_traversal_frames = 0
        self._is_traversing = False
        self._target_node = None
        self._topology_id = -1
        self._halt_frames = 0
        self._mandatory_wait_time = 2
        self._starboard_turn_frames = 0

    def evaluate_traversal_safety(self, local_planner, preview_steps):
        """
        Valuta se è sicuro impegnare e attraversare l'incrocio.
        La logica impone prima un tempo di sosta obbligatorio, per poi analizzare
        topologicamente l'incrocio e le traiettorie dei veicoli che lo occupano.

        Args:
            local_planner (LocalPlanner): Il pianificatore locale per ottenere i waypoint futuri.
            preview_steps (int): Numero di passi in avanti per ottenere la direzione futura.

        Returns:
            bool: True se l'incrocio è libero e sicuro da attraversare, False altrimenti.
        """
        self._local_planner = local_planner
        _, routing_directive = self._local_planner.get_incoming_waypoint_and_direction(steps=preview_steps)

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

        ahead_node = spatial_nodes['front'][0] if spatial_nodes['front'] else None
        port_node = spatial_nodes['left'][0] if spatial_nodes['left'] else None
        starboard_node = spatial_nodes['right'][0] if spatial_nodes['right'] else None

        v_ahead = self._get_ordered_vehicles(ahead_node, 9)[0] if ahead_node and self._get_ordered_vehicles(ahead_node,                                                                                              9) else None
        v_port = self._get_ordered_vehicles(port_node, 9)[0] if port_node and self._get_ordered_vehicles(port_node,                                                                            9) else None
        v_starboard = self._get_ordered_vehicles(starboard_node, 9)[0] if starboard_node and self._get_ordered_vehicles(starboard_node, 9) else None

        if self._topology_id == IntersectionTopology.AHEAD_PORT:
            if not v_ahead and not v_port: return True
            if routing_directive == RoadOption.LEFT:
                if not self._check_linear_trajectory(v_ahead) and not self._check_starboard_signal(v_ahead) and not self._check_port_signal(v_port): return True
            elif routing_directive == RoadOption.STRAIGHT:
                if not self._check_port_signal(v_port): return True
            return False

        elif self._topology_id == IntersectionTopology.AHEAD_STARBOARD:
            if not v_ahead and not v_starboard: return True
            if routing_directive == RoadOption.RIGHT:
                if not self._check_port_signal(v_ahead) or (self._check_port_signal(v_starboard) and get_speed(v_starboard) > 0.1): return True
            elif routing_directive == RoadOption.STRAIGHT:
                if not self._check_port_signal(v_starboard) and not self._check_starboard_signal(v_starboard) and not self._check_port_signal(v_ahead): return True
            return False

        elif self._topology_id == IntersectionTopology.PORT_STARBOARD:
            if not v_port and not v_starboard: return True
            if routing_directive == RoadOption.LEFT:
                if not self._check_linear_trajectory(v_port) and not self._check_linear_trajectory(v_starboard) and not self._check_port_signal(v_starboard):
                    return True
                elif self._check_starboard_signal(v_port) and v_starboard in occupying_fleet:
                    return True
            elif routing_directive == RoadOption.RIGHT:
                if not self._check_linear_trajectory(v_port): return True
            return False

        elif self._topology_id == IntersectionTopology.OMNIDIRECTIONAL:
            if not v_ahead and not v_port and not v_starboard: return True
            if routing_directive == RoadOption.STRAIGHT:
                if not self._check_port_signal(v_port) and not self._check_starboard_signal(v_starboard) and not self._check_port_signal(v_ahead) \
                        and not self._check_linear_trajectory(v_port) and not self._check_linear_trajectory(v_starboard):
                    return True
                elif self._check_starboard_signal(v_port) and v_port in occupying_fleet:
                    return True
                elif (self._check_linear_trajectory(v_ahead) or self._check_starboard_signal(v_ahead)) and v_ahead in occupying_fleet:
                    return True
            elif routing_directive == RoadOption.LEFT:
                if not self._check_linear_trajectory(v_ahead) and not self._check_linear_trajectory(v_port) and not self._check_linear_trajectory(v_starboard) \
                        and not self._check_starboard_signal(v_ahead) and not self._check_port_signal(v_port) and not self._check_port_signal(v_starboard):
                    return True
                elif (self._check_port_signal(v_ahead) or self._check_starboard_signal(v_ahead)) and v_ahead in occupying_fleet:
                    return True
                elif self._check_starboard_signal(v_port) and v_port in occupying_fleet:
                    return True
                elif self._check_starboard_signal(v_starboard) and v_starboard in occupying_fleet:
                    return True
            elif routing_directive == RoadOption.RIGHT:
                if not self._check_linear_trajectory(v_port) and not self._check_port_signal(v_ahead): return True
            return False

        return False

    @property
    def is_traversing(self):
        """
        Proprietà che indica se il veicolo sta attualmente attraversando un incrocio.

        Returns:
            bool: Stato di attraversamento attuale.
        """
        return self._is_traversing

    @is_traversing.setter
    def is_traversing(self, val):
        """
        Imposta lo stato di attraversamento dell'incrocio.

        Args:
            val (bool): Nuovo stato di attraversamento.

        Returns:
            None
        """
        self._is_traversing = val

    @property
    def active_traversal_frames(self):
        """
        Proprietà che restituisce il numero di frame attivi rimanenti per l'attraversamento.

        Returns:
            int: Numero di frame.
        """
        return self._active_traversal_frames

    @active_traversal_frames.setter
    def active_traversal_frames(self, val):
        """
        Imposta il numero di frame per l'attraversamento attivo.

        Args:
            val (int): Numero di frame desiderati.

        Returns:
            None
        """
        self._active_traversal_frames = val

    @property
    def halt_frames(self):
        """
        Proprietà che restituisce il numero di frame in cui il veicolo è stato fermo all'incrocio.

        Returns:
            int: Frame di sosta.
        """
        return self._halt_frames

    @halt_frames.setter
    def halt_frames(self, val):
        """
        Imposta il contatore dei frame di sosta all'incrocio.

        Args:
            val (int): Numero di frame di sosta.

        Returns:
            None
        """
        self._halt_frames = val

    @property
    def topology_id(self):
        """
        Proprietà che restituisce l'ID topologico dell'incrocio corrente.

        Returns:
            int: Valore corrispondente a IntersectionTopology.
        """
        return self._topology_id

    def _map_entry_nodes(self, intersection):
        """
        Estrae i nodi (waypoint) di ingresso per un dato incrocio filtrando i duplicati.

        Args:
            intersection (carla.Junction): L'incrocio da analizzare.

        Returns:
            set: Insieme di carla.Waypoint che rappresentano gli ingressi unici dell'incrocio.
        """
        junction_wps = intersection.get_waypoints(carla.LaneType.Driving)
        entry_wps = set()
        seen_coords = set()
        for begin, _ in junction_wps:
            key = (round(begin.transform.location.x, 2), round(begin.transform.location.y, 2), round(begin.transform.location.z, 2))
            if key in seen_coords: continue
            entry_wps.add(begin)
            seen_coords.add(key)
        return entry_wps

    def _categorize_spatial_nodes(self, point, neighbors, direction=None):
        """
        Classifica i nodi vicini (avanti, dietro, sinistra, destra) rispetto a un punto di riferimento,
        utilizzando il prodotto scalare e vettoriale per determinare la posizione relativa.

        Args:
            point (carla.Transform): Il punto di riferimento (pivot) e orientamento.
            neighbors (list): Lista di carla.Waypoint o carla.Transform da classificare.
            direction (str, opzionale): Se specificato ('left', 'right', 'front', 'back'), restituisce solo quella lista.

        Returns:
            dict o list: Dizionario con le liste dei nodi suddivise per direzione, oppure la singola lista se direction è specificato.
        """
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
        """
        Classifica la topologia dell'incrocio determinando quali direzioni (sinistra, destra, dritto)
        sono disponibili a partire dal waypoint del veicolo.

        Args:
            vehicle_wp (carla.Waypoint): Il waypoint corrente del veicolo in prossimità dell'incrocio.
            intersection (carla.Junction): L'incrocio in questione.

        Returns:
            IntersectionTopology: La classificazione della topologia dell'incrocio.
        """
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

    @staticmethod
    def _check_linear_trajectory(vehicle):
        """
        Verifica se un veicolo sta procedendo dritto controllando lo stato dei suoi indicatori di direzione.

        Args:
            vehicle (carla.Vehicle): Il veicolo da esaminare.

        Returns:
            bool: True se nessun indicatore di direzione è attivo (traiettoria lineare), False altrimenti.
        """
        if not vehicle: return False
        light_state = vehicle.get_light_state()
        if bool(light_state & carla.libcarla.VehicleLightState.LeftBlinker) or bool(
            light_state & carla.libcarla.VehicleLightState.RightBlinker): return False
        return True

    @staticmethod
    def _check_port_signal(vehicle):
        """
        Verifica se l'indicatore di direzione sinistro di un veicolo è attivo.

        Args:
            vehicle (carla.Vehicle): Il veicolo da esaminare.

        Returns:
            bool: True se la freccia sinistra è accesa, False altrimenti.
        """
        if not vehicle: return False
        return bool(vehicle.get_light_state() & carla.libcarla.VehicleLightState.LeftBlinker)

    @staticmethod
    def _check_starboard_signal(vehicle):
        """
        Verifica se l'indicatore di direzione destro di un veicolo è attivo.

        Args:
            vehicle (carla.Vehicle): Il veicolo da esaminare.

        Returns:
            bool: True se la freccia destra è accesa, False altrimenti.
        """
        if not vehicle: return False
        return bool(vehicle.get_light_state() & carla.libcarla.VehicleLightState.RightBlinker)

    @property
    def starboard_turn_frames(self):
        """
        Proprietà che restituisce il numero di frame rimanenti per la gestione di una svolta a destra.

        Returns:
            int: Numero di frame per la svolta a dritta.
        """
        return self._starboard_turn_frames

    @starboard_turn_frames.setter
    def starboard_turn_frames(self, val):
        """
        Imposta il numero di frame per la gestione della svolta a destra.

        Args:
            val (int): Numero di frame desiderato.

        Returns:
            None
        """
        self._starboard_turn_frames = val

    def execute_traversal(self, routing_directive, lock_frames = 120):
        """
        Inizia e gestisce l'impegno dell'incrocio, bloccando lo stato per un numero specificato di frame.
        In caso di svolta a destra in incroci omnidirezionali, applica un offset laterale.

        Args:
            routing_directive (RoadOption): La direzione prevista che il veicolo deve prendere.
            lock_frames (float, opzionale): Il numero di frame per cui bloccare lo stato di attraversamento. Default a 120.

        Returns:
            carla.VehicleControl: Il comando di controllo generato dal local planner per il veicolo.
        """
        self._active_traversal_frames = lock_frames
        self._is_traversing = True

        if routing_directive == RoadOption.RIGHT and self._topology_id == IntersectionTopology.OMNIDIRECTIONAL:
            self._starboard_turn_frames = round(2 / self._world.get_snapshot().timestamp.delta_seconds)
            self._local_planner.set_lateral_offset(-0.3)

        return self._local_planner.run_step()