# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import carla
from shapely.geometry import Polygon, MultiPoint

from local_planner import LocalPlanner, RoadOption
from global_route_planner import GlobalRoutePlanner
from misc import *

class BasicAgent(object):
    """
    Agente di base per la navigazione autonoma in CARLA.

    Questa classe fornisce le funzionalità fondamentali per far muovere un veicolo
    da un punto A a un punto B, gestendo la pianificazione globale e locale,
    e implementando reazioni base a ostacoli, semafori e segnali di stop.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Inizializza il BasicAgent.

        Args:
            vehicle (carla.Vehicle): L'attore veicolo da controllare.
            opt_dict (dict, opzionale): Dizionario di configurazione con parametri per l'agente.
            map_inst (carla.Map, opzionale): Istanza della mappa di CARLA. Se non fornita, viene recuperata dal mondo.
            grp_inst (GlobalRoutePlanner, opzionale): Istanza preesistente del Global Planner.
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None

        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = opt_dict.get('base_tlight_threshold', 5.0)
        self._base_sign_threshold = opt_dict.get('base_sign_threshold', 20.0)
        self._base_vehicle_threshold = opt_dict.get('base_vehicle_threshold', 5.0)
        self._speed_ratio = opt_dict.get('detection_speed_ratio', 1.0)
        self._pedestrian_scan_normal = opt_dict.get('pedestrian_scan_normal', 20.0)
        self._pedestrian_scan_junction = opt_dict.get('pedestrian_scan_junction', 12.0)
        self._obstacle_scan_radius = opt_dict.get('obstacle_scan_radius', 20.0)
        self._obstacle_critical_dist = opt_dict.get('obstacle_critical_dist', 6.0)
        self._fleet_straight_tolerance = opt_dict.get('fleet_straight_tolerance', 10.0)
        self._fleet_center_offset = opt_dict.get('fleet_center_offset', 0.3)
        self._max_brake = 0.5
        self._offset = 0
        self._simulation_timestamp = 0.05

        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']

        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}

    def add_emergency_stop(self, control):
        """
        Modifica un oggetto di controllo per applicare una frenata di emergenza.
        Azzera l'acceleratore e applica il freno massimo configurato.

        Args:
            control (carla.VehicleControl): Il controllo veicolo corrente.

        Returns:
            carla.VehicleControl: Il controllo modificato con la frenata di emergenza.
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Imposta la velocità target per l'agente e aggiorna il local planner.

        Args:
            speed (float): La nuova velocità target in km/h.
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        Attiva o disattiva l'adattamento ai limiti di velocità stradali.

        Args:
            value (bool, opzionale): True per seguire i limiti, False per ignorarli.
                                     Default è True.
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """
        Recupera l'istanza del LocalPlanner in uso.
        Returns:
            LocalPlanner: Il pianificatore locale dell'agente.
        """
        return self._local_planner

    def get_global_planner(self):
        """
        Recupera l'istanza del GlobalRoutePlanner in uso.
        Returns:
            GlobalRoutePlanner: Il pianificatore globale dell'agente.
        """
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        Genera un percorso dal punto di partenza a quello di destinazione e lo assegna.

        Se `start_location` non è fornito, il percorso inizierà dal waypoint
        attualmente tracciato dal planner locale.

        Args:
            end_location (carla.Location): Le coordinate di destinazione.
            start_location (carla.Location, opzionale): Le coordinate di partenza.
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Assegna un piano globale pre-calcolato (lista di waypoint) al planner locale.
        Args:
            plan (list): Lista di tuple (carla.Waypoint, RoadOption) che formano la rotta.
            stop_waypoint_creation (bool, opzionale): Ferma la generazione automatica di waypoint.
            clean_queue (bool, opzionale): Svuota la coda corrente di waypoint prima di applicare il piano.
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calcola il percorso tra due waypoint utilizzando il planner globale.
        Args:
            start_waypoint (carla.Waypoint): Waypoint di partenza.
            end_waypoint (carla.Waypoint): Waypoint di destinazione.
        Returns:
            list: Lista di tuple (carla.Waypoint, RoadOption) che rappresentano il percorso.
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """
        Esegue un passo della logica di controllo dell'agente.

        Rileva potenziali pericoli (veicoli ostacolo, semafori rossi, segnali di stop)
        calcolando distanze dinamiche basate sulla velocità. Se rileva un pericolo,
        applica una frenata di emergenza; altrimenti, restituisce i controlli normali
        del planner locale.

        Returns:
            carla.VehicleControl: Il comando da applicare al veicolo in questo tick.
        """
        hazard_detected = False

        vehicle_list = self._world.get_actors().filter("*vehicle*")

        vehicle_speed = get_speed(self._vehicle) / 3.6

        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._vehicle, self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        max_sign_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_sign, _ = self._affected_by_sign(self._vehicle, max_sign_distance, sign_type="206")
        if affected_by_sign:
            hazard_detected = True

        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control

    def reset(self):
        """
        Reimposta lo stato dell'agente. (Attualmente non implementato nel BasicAgent).
        """
        pass

    def done(self):
        """
        Verifica se l'agente ha raggiunto la sua destinazione finale.
        Returns:
            bool: True se il percorso è completato, False altrimenti.
        """
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """
        Imposta se l'agente deve ignorare i semafori o rispettarli.
        Args:
            active (bool, opzionale): True per ignorare i semafori, False per rispettarli.
        """
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """
        Imposta se l'agente deve ignorare i segnali di STOP o rispettarli.
        Args:
            active (bool, opzionale): True per ignorare i segnali di STOP, False per rispettarli.
        """
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """
        Imposta se l'agente deve ignorare gli altri veicoli o no.
        Args:
            active (bool, opzionale): True per ignorare i veicoli, False per considerarli.
        """
        self._ignore_vehicles = active

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=5):
        """
        Pianifica ed esegue una manovra di cambio corsia.
        Genera un nuovo percorso locale che forza lo spostamento laterale.

        Args:
            direction (str o enum): Direzione del cambio corsia (es. 'left' o 'right').
            same_lane_time (float, opzionale): Tempo in secondi da percorrere nella corsia corrente prima della manovra.
            other_lane_time (float, opzionale): Tempo in secondi da percorrere nella nuova corsia.
            lane_change_time (float, opzionale): Tempo stimato per eseguire la transizione.
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light(self, vehicle, lights_list = None, max_distance = None):
        """
        Verifica se il veicolo è influenzato da un semaforo rosso rilevante lungo la sua traiettoria.
        Analizza la posizione dei trigger dei semafori, l'orientamento della strada
        e la distanza per determinare se è necessario fermarsi.

        Args:
            vehicle (carla.Vehicle): Il veicolo da analizzare.
            lights_list (list, opzionale): Lista degli attori semaforo. Se None, vengono recuperati dal mondo.
            max_distance (float, opzionale): Distanza massima entro cui considerare il semaforo.

        Returns:
            tuple: (bool, carla.TrafficLight o None)
                - bool: True se il veicolo deve fermarsi per un semaforo rosso, False altrimenti.
                - carla.TrafficLight o None: L'oggetto semaforo rilevato, None se non c'è restrizione.
        """

        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _affected_by_sign(self, vehicle, sign_type = "206", max_distance = None):
        """
        Verifica se il veicolo è influenzato da un segnale stradale specifico (es. STOP)
        nella sua corsia attuale.

        Args:
            vehicle (carla.Vehicle): Il veicolo da analizzare.
            sign_type (str, opzionale): L'ID del tipo di segnale in OpenDRIVE (es. "206" per STOP).
            max_distance (float, opzionale): Distanza massima di rilevamento.

        Returns:
            tuple: (bool, carla.Landmark o None)
                - bool: True se il segnale influenza il veicolo, False altrimenti.
                - carla.Landmark o None: L'oggetto segnale rilevato, o None se non trovato.
        """
        if self._ignore_stop_signs:
            return False, None

        if max_distance is None:
            max_distance = self._base_sign_threshold

        target_vehicle_wp = self._map.get_waypoint(vehicle.get_location())
        signs_list = target_vehicle_wp.get_landmarks_of_type(max_distance, type=sign_type, stop_at_junction=False)

        if sign_type == '206':
            signs_list = [sign for sign in signs_list if
                          self._map.get_waypoint(vehicle.get_location()).lane_id == sign.waypoint.lane_id]

            if signs_list:
                return True, signs_list[0]
        return False, None

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, lane_offset=0):
        """
        Rileva la presenza di veicoli o pedoni ostacolo calcolando l'intersezione tra il poligono
        spaziale del percorso futuro dell'ego vehicle e le bounding box degli altri attori.

        Questa versione ottimizzata costruisce geometricamente un corridoio 2D valido che
        rappresenta la traiettoria spazzata dall'auto e utilizza la convex hull per
        proiettare l'impronta a terra degli ostacoli. Ciò garantisce il rilevamento
        preciso anche di attori "accartocciati" o sdraiati a terra (es. pedoni caduti).

        Args:
            vehicle_list (list, opzionale): Lista degli attori (veicoli/pedoni) da esaminare.
                                            Se None, recupera tutti i veicoli nella scena.
            max_distance (float, opzionale): Distanza massima entro cui calcolare e cercare pericoli.
                                             Se None, utilizza la soglia di base del comportamento.
            lane_offset (int, opzionale): Variazione della corsia in cui proiettare il corridoio
                                          di ricerca (es. -1 per cercare ostacoli nella corsia di sinistra).

        Returns:
            tuple: (bool, carla.Actor o None, float) Rispettivamente indicatore di pericolo (True se presente),
                   l'attore target rilevato e la distanza spaziale 3D da esso (-1 se la via è libera).
        """
        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        right_points = []
        left_points = []

        ego_location = ego_transform.location
        extent_y = self._vehicle.bounding_box.extent.y
        r_vec = ego_transform.get_right_vector()

        p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
        p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
        right_points.append([p1.x, p1.y, p1.z])
        left_points.append([p2.x, p2.y, p2.z])

        for wp, _ in self._local_planner.get_plan():
            if ego_location.distance(wp.transform.location) > max_distance:
                break
            r_vec = wp.transform.get_right_vector()
            p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
            p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
            right_points.append([p1.x, p1.y, p1.z])
            left_points.append([p2.x, p2.y, p2.z])

        if len(right_points) + len(left_points) < 3:
            return (False, None, -1)

        route_bb = right_points + left_points[::-1]
        ego_polygon = Polygon(route_bb)

        if not ego_polygon.is_valid:
            ego_polygon = ego_polygon.buffer(0)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_location = target_vehicle.get_location()

            z_diff = abs(ego_location.z - target_location.z)
            if z_diff >= 2.5:
                continue

            if ego_location.distance(target_location) > max_distance:
                continue

            target_bb = target_vehicle.bounding_box
            target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
            target_list = [[v.x, v.y, v.z] for v in target_vertices]

            target_polygon = MultiPoint(target_list).convex_hull

            if ego_polygon.intersects(target_polygon):
                return (True, target_vehicle, compute_distance(target_location, ego_location))

        return (False, None, -1)

    def _generate_lane_change_path(self, waypoint, direction = 'left', distance_same_lane = 10, distance_other_lane = 25, lane_change_distance = 25,
                                   check = True, lane_changes = 1, step_distance = 4.5, concorde = False):

        """
        Genera una sequenza di waypoint per eseguire una manovra di cambio corsia.

        Calcola i waypoint necessari per rimanere sulla corsia corrente, effettuare
        la transizione laterale e infine stabilizzarsi sulla nuova corsia.

        Args:
            waypoint (carla.Waypoint): Waypoint iniziale da cui calcolare la manovra.
            direction (str, opzionale): 'left' o 'right'. Default è 'left'.
            distance_same_lane (float, opzionale): Metri da percorrere nella corsia attuale prima di iniziare il cambio.
            distance_other_lane (float, opzionale): Metri da percorrere nella nuova corsia.
            lane_change_distance (float, opzionale): Distanza longitudinale per completare la transizione.
            check (bool, opzionale): Se True, verifica la legalità/fattibilità del cambio corsia nella topologia stradale.
            lane_changes (int, opzionale): Numero di corsie da attraversare.
            step_distance (float, opzionale): Distanza tra waypoint consecutivi.
            concorde (bool, opzionale): Se True, garantisce che il calcolo sulla nuova corsia segua il flusso di traffico concorde in base al lane_id.

        Returns:
            list: Una lista di tuple (carla.Waypoint, RoadOption) che costituisce il piano
                  locale temporaneo, oppure una lista vuota `[]` se la manovra è impossibile.
        """
        plan = [(waypoint, RoadOption.LANEFOLLOW)]
        option = RoadOption.LANEFOLLOW

        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        while lane_changes_done < lane_changes:

            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            plan.append((side_wp, option))
            lane_changes_done += 1

        distance = 0
        pivot = plan[-1][0].lane_id
        while distance < distance_other_lane:
            if waypoint.lane_id * pivot > 0 and concorde:
                next_wps = plan[-1][0].next(step_distance)
            else:
                next_wps = plan[-1][0].previous(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan

    def _get_ordered_vehicles(self, reference, max_distance):
        """
        Restituisce una lista di veicoli nell'ambiente, ordinata per vicinanza al riferimento.
        Esclude i veicoli eccessivamente vicini (< 0.1m, spesso l'ego stesso o collisioni)
        e quelli oltre la distanza massima consentita.

        Args:
            reference (carla.Actor o carla.Location): L'entità o punto rispetto a cui calcolare la distanza.
            max_distance (float): La distanza di ricerca massima.

        Returns:
            list: Lista di oggetti `carla.Vehicle` ordinata dal più vicino al più lontano.
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        if isinstance(reference, carla.Actor):
            vehicle_list = [ v for v in vehicle_list if v.id != reference.id and 0.1 < get_distance(v, reference) < max_distance ]
        else:
            vehicle_list = [
                v for v in vehicle_list
                if 0.1 < get_distance(v, reference) < max_distance
            ]

        vehicle_list.sort(key=lambda v: get_distance(v, reference))
        return vehicle_list

    def _parked_vehicle(self, vehicle):
        """
        Verifica se un determinato veicolo target è parcheggiato (o permanentemente bloccato).

        Un veicolo è considerato parcheggiato se la sua velocità è quasi nulla (< 0.1), non
        si trova in prossimità di incroci, non sta aspettando un semaforo, non ha uno STOP
        e non è bloccato in coda a causa di veicoli davanti a lui che stanno rispettando
        la segnaletica.

        Args:
            vehicle (carla.Vehicle): Il veicolo target da analizzare.

        Returns:
            tuple: (carla.Waypoint, bool)
                - carla.Waypoint: Il waypoint su cui si trova il veicolo analizzato.
                - bool: True se il veicolo è considerato "parcheggiato" o "ostacolo fisso",
                        False se è nel traffico dinamico.
        """
        vehicle_loc = vehicle.get_location()
        vehicle_wp = self._map.get_waypoint(vehicle_loc)

        lights_list = self._world.get_actors().filter("*traffic_light*")
        lights_list = [l for l in lights_list if is_within_distance(l.get_transform(), vehicle.get_transform(), 50, angle_interval=[0, 90])]

        affected_by_traffic_light, _ = self._affected_by_traffic_light(self._vehicle)
        affected_by_stop_sign, _ = self._affected_by_sign(self._vehicle)

        if not affected_by_stop_sign or not affected_by_traffic_light:
            vehicle_list = self._get_ordered_vehicles(vehicle, 30)
            for v in vehicle_list:
                if v.id == self._vehicle.id:
                    continue
                affected_by_traffic_light, _ = self._affected_by_traffic_light(v)
                affected_by_stop_sign, _ = self._affected_by_sign(v)
                if affected_by_stop_sign or affected_by_traffic_light:
                    break

        if get_speed(vehicle) < 0.1 and not affected_by_stop_sign and not affected_by_traffic_light and not vehicle_wp.is_junction and not lights_list:
            return vehicle_wp, True

        return vehicle_wp, False