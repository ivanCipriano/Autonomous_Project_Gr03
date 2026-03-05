# cognitive_modules/base_evaluator.py
import carla
import numpy as np
from misc import is_within_distance, get_distance, get_speed
from local_planner import RoadOption


class BaseEvaluator:
    """
    Interfaccia base per la catena di valutazione cognitiva e libreria di funzioni percettive.

    Questa classe funge da architettura fondamentale per i moduli decisionali di un veicolo 
    a guida autonoma all'interno del simulatore CARLA. Fornisce i metodi essenziali per 
    l'analisi dell'ambiente circostante, l'interazione con gli altri attori e la gestione 
    della cinematica longitudinale (come l'Adaptive Cruise Control) e le logiche di evasione 
    degli ostacoli.
    """

    def __init__(self, core_system):
        """
        Inizializza il modulo di valutazione di base.

        Collega l'istanza corrente al sistema di controllo centrale (core system) dell'agente 
        autonomo, permettendo l'accesso allo stato del veicolo, ai parametri comportamentali e 
        ai dati aggregati dei sensori.

        Args:
            core_system (object): L'istanza del sistema centrale o dell'agente principale 
                                che gestisce il ciclo di vita e lo stato globale del veicolo.
        """
        self.core_system = core_system

    def evaluate(self, **kwargs) -> carla.VehicleControl:
        """
        Valuta lo stato corrente e calcola i comandi di controllo del veicolo.

        Questo è un metodo astratto che deve essere obbligatoriamente implementato dalle 
        sottoclassi per definire specifiche euristiche o modelli decisionali (es. reti neurali, 
        alberi comportamentali) per la presa di decisione tattica o operativa.

        Args:
            **kwargs: Argomenti variabili dipendenti dalla specifica implementazione del modulo.

        Returns:
            carla.VehicleControl: I comandi di accelerazione, frenata e sterzata da applicare.

        Raises:
            NotImplementedError: Se il metodo non viene sovrascritto dalla classe derivata.
        """
        raise NotImplementedError("Il metodo evaluate deve essere implementato.")

    def halt_vehicle(self):
        """
        Genera un comando di controllo per l'arresto immediato del veicolo autonomo.

        Applica la massima pressione sul freno (basata sui limiti fisici o imposti e configurati 
        nel core_system) e azzera l'acceleratore, garantendo un arresto in sicurezza senza 
        l'attivazione del freno a stazionamento. Viene tipicamente invocato in situazioni di 
        emergenza o in presenza di un ostacolo critico a distanza ravvicinata.

        Returns:
            carla.VehicleControl: Comando di controllo con acceleratore a 0.0 e freno impostato 
                                al valore massimo consentito.
        """
        cmd = carla.VehicleControl()
        cmd.throttle = 0.0
        cmd.brake = self.core_system._max_brake
        cmd.hand_brake = False
        return cmd

    def adaptive_cruise_control(self, target_vehicle, distance, debug=False):
        """
        Modula la velocità longitudinale in base al veicolo che precede (Car Following).

        Implementa la logica dell'Adaptive Cruise Control (ACC) calcolando il Time-To-Collision 
        (TTC) in base al delta di velocità. Adegua la velocità target dell'ego-vehicle 
        (accelerando, mantenendo l'andatura o frenando dolcemente) in base alla distanza 
        relativa, nel rigoroso rispetto dei limiti di velocità della mappa e dei parametri di 
        sicurezza impostati nel profilo comportamentale.

        Args:
            target_vehicle (carla.Actor): Il veicolo rilevato dinanzi all'ego-vehicle.
            distance (float): La distanza in metri tra l'ego-vehicle e il target_vehicle.
            debug (bool, opzionale): Se True, attiva il tracciamento grafico di debug 
                                    all'interno del planner locale. Default a False.

        Returns:
            carla.VehicleControl: Il comando di attuazione ottimizzato dal controller PID 
                                per raggiungere la velocità target di sicurezza.
        """
        sys = self.core_system
        vehicle_speed = get_speed(target_vehicle)
        delta_v = max(1, (sys._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        if sys._behavior.safety_time > ttc > 0.0:
            target_speed = min([self.positive(vehicle_speed - sys._behavior.speed_decrease), sys._behavior.max_speed,
                                sys._speed_limit - sys._behavior.speed_lim_dist])
        elif 2 * sys._behavior.safety_time > ttc >= sys._behavior.safety_time:
            target_speed = min([max(sys._min_speed, vehicle_speed), sys._behavior.max_speed,
                                sys._speed_limit - sys._behavior.speed_lim_dist])
        else:
            target_speed = min([sys._behavior.max_speed, sys._speed_limit - sys._behavior.speed_lim_dist])

        sys._local_planner.set_speed(target_speed)
        return sys._local_planner.run_step(debug=debug)

    def _process_tailgating(self, waypoint, vehicle_list):
        """
        Gestisce le dinamiche di inseguimento ravvicinato e valuta manovre evasive di cambio corsia.

        Analizza la presenza di veicoli più lenti immediatamente davanti all'ego-vehicle. Se 
        le condizioni di viabilità lo permettono (es. linee di demarcazione tratteggiate) e 
        le corsie adiacenti sono libere da ostacoli, il metodo innesca proattivamente una manovra 
        di sorpasso verso la corsia di destra o sinistra per risolvere la situazione di congestione 
        (tailgating resolution).

        Args:
            waypoint (carla.Waypoint): Il waypoint corrente su cui si trova l'ego-vehicle.
            vehicle_list (list): Lista degli attori di tipo veicolo presenti nell'ambiente simulato.
        """
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
        """
        Scansiona l'ambiente circostante per rilevare flotte di veicoli o ciclisti.

        Filtra gli attori nella simulazione considerando solo quelli in un raggio spaziale di 
        interesse. A seconda dell'intenzione di manovra dell'ego-vehicle (mantenimento corsia, 
        cambio corsia a sinistra o destra), il metodo controlla eventuali ostacoli frontali 
        o laterali calcolando offset di corsia, tolleranze angolari e proiettando bounding box. 
        Include logiche specifiche di alta priorità per l'identificazione e la salvaguardia dei ciclisti.

        Args:
            waypoint (carla.Waypoint): Il waypoint corrente dell'ego-vehicle.

        Returns:
            tuple: Una tupla contenente tre elementi:
                - bool: True se è stato rilevato un ostacolo rilevante, False altrimenti.
                - carla.Actor o None: L'istanza dell'oggetto veicolo/ciclista rilevato come ostacolo.
                - float: La distanza in metri dall'ostacolo, oppure -1 se l'area è libera.
        """
        sys = self.core_system
        v_list = sys._world.get_actors().filter("*vehicle*")
        v_list = [v for v in v_list if get_distance(v, waypoint) < 13 and v.id != sys._vehicle.id]

        if not v_list:
            return False, None, -1

        bicycle_list = [b for b in v_list if
                        self.is_a_bicycle(b.type_id) and is_within_distance(b.get_transform(), sys._vehicle.get_transform(),
                                                                       10, angle_interval=[0, 90])]
        if len(bicycle_list) == 1:
            print('[Cognition] -> Cyclist track intercepted.')
            return True, bicycle_list[0], get_distance(bicycle_list[0], waypoint)

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

    def is_a_bicycle(self,vehicle_name):
        """
        Verifica se l'identificativo del veicolo corrisponde a una bicicletta.

        Funzione di classificazione semantica per il modulo percettivo. Distingue i velocipedi 
        dai normali veicoli a motore confrontando il type_id di CARLA con una whitelist di modelli, 
        permettendo al sistema di adottare cautele specifiche per gli utenti vulnerabili della strada.

        Args:
            vehicle_name (str): L'identificativo testuale del modello dell'attore (es. 'vehicle.bh.crossbike').

        Returns:
            bool: True se il modello appartiene alla categoria delle biciclette, False altrimenti.
        """
        BICYCLES = ['vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']
        return vehicle_name in BICYCLES

    def positive(self,num):
        """
        Garantisce la restituzione di un valore scalare strettamente non negativo.

        Funzione matematica di supporto che agisce come un rettificatore. Viene impiegata 
        frequentemente nei calcoli di controllo longitudinale (es. limitazione della velocità 
        o delta di decelerazione) per neutralizzare anomalie matematiche e prevenire richieste di 
        velocità negative non fisicamente attuabili.

        Args:
            num (float): Il valore numerico in ingresso da normalizzare.

        Returns:
            float: Il valore originale se è maggiore di 0.0, altrimenti 0.0.
        """
        return num if num > 0.0 else 0.0