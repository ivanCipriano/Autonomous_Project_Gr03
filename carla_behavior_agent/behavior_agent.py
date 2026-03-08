# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from tactical_modules.trajectory_bypass_engine import TrajectoryBypassEngine
from tactical_modules.intersection_navigation_engine import IntersectionNavigationEngine
from cognitive_modules.signal_evaluator import TrafficSignalEvaluator
from cognitive_modules.pedestrian_evaluator import BipedalHazardEvaluator
from cognitive_modules.obstacle_evaluator import StaticObstructionEvaluator
from cognitive_modules.stop_sign_evaluator import StopEvaluator
from cognitive_modules.fleet_evaluator import FleetProximityEvaluator
from cognitive_modules.cruise_evaluator import NavigationCruiseEvaluator
from misc import *
from types import SimpleNamespace

class BehaviorAgent(BasicAgent):
    """
    Agente autonomo avanzato per la navigazione e il raggiungimento di una destinazione.
    Integra un'architettura decisionale basata su una "Chain of Responsibility" (catena di valutatori cognitivi)
    e su motori tattici per la gestione di incroci e manovre di elusione/sorpasso.
    Supporta profili comportamentali dinamici (Cauto, Normale, Aggressivo) e adatta la guida
    alle condizioni meteorologiche e ai limiti di velocità.
    """

    def __init__(self, vehicle, behavior='cautious', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Inizializza l'agente comportamentale, configurando i parametri di base, il profilo di guida
        e i vari moduli cognitivi e tattici necessari per l'analisi dell'ambiente.

        Args:
            vehicle (carla.Vehicle): L'attore veicolo da controllare nel simulatore.
            behavior (str, opzionale): Il tipo di comportamento da adottare ('cautious', 'normal', 'aggressive'). Default a 'cautious'.
            opt_dict (dict, opzionale): Dizionario per configurazioni aggiuntive dell'agente. Default a {}.
            map_inst (carla.Map, opzionale): L'istanza della mappa per le interrogazioni spaziali. Default a None.
            grp_inst (GlobalRoutePlanner, opzionale): Il pianificatore di rotte globali. Default a None.

        Returns:
            None
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)

        self._approach_speed = opt_dict.get('approach_speed', 10.0)
        self._min_speed = opt_dict.get('min_speed', 5.0)
        self._rain_speed_penalty = opt_dict.get('rain_speed_penalty', 5.0)
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._look_ahead_steps = 0
        self._speed = 0
        self._speed_limit = 0
        self._stuck = False

        self._bypass_engine = TrajectoryBypassEngine(self._vehicle, opt_dict)
        self._evaluators_chain = [
            TrafficSignalEvaluator(core_system=self),
            BipedalHazardEvaluator(core_system=self),
            StaticObstructionEvaluator(core_system=self),
            StopEvaluator(core_system=self),
            FleetProximityEvaluator(core_system=self),
            NavigationCruiseEvaluator(core_system=self)
        ]

        self._bypass_engine = TrajectoryBypassEngine(self._vehicle, opt_dict)
        self._navigation_engine = IntersectionNavigationEngine(self._vehicle, opt_dict)

        profiles = opt_dict.get('behavior_profiles', {})
        if behavior in profiles:
            self._behavior = SimpleNamespace(**profiles[behavior])
        else:
            self._behavior = Cautious()
        print(f"[INIT] Behavior profile set to: {behavior.capitalize()}")

        self._direction = None
        self._is_raining = False
        
        # Log collisioni
        blueprint_library = self._world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        self._collision_sensor = self._world.spawn_actor(collision_bp, carla.Transform(), attach_to=self._vehicle)
        self._collision_sensor.listen(lambda event: self._on_collision(event))

    def run_step(self, debug=False):
        """
        Ciclo di esecuzione principale dell'agente.
        Aggiorna le informazioni di contesto e interroga sequenzialmente la catena dei valutatori
        cognitivi. Ritorna l'azione (comando di controllo) stabilita dal primo valutatore che rileva
        una condizione critica o dalla navigazione di crociera di default.

        Args:
            debug (bool, opzionale): Flag per l'attivazione della stampa di log di debug. Default a False.

        Returns:
            carla.VehicleControl: Il comando di controllo generato (acceleratore, freno, sterzo, ecc.).
        """
        self.__update_information(debug=debug)

        environment_context = {
            'ego_vehicle_wp': self._map.get_waypoint(self._vehicle.get_location()),
            'debug': debug
        }

        for evaluator in self._evaluators_chain:
            action = evaluator.evaluate(**environment_context)
            if action is not None:
                return action

        return carla.VehicleControl()


    def __normal_behaviour(self, debug = False):
        """
        Gestisce e applica il comportamento di guida standard (non in emergenza o in manovre speciali).
        Calcola la velocità target in base al limite di velocità corrente e al profilo comportamentale,
        delegando poi al pianificatore locale la generazione del comando.

        Args:
            debug (bool, opzionale): Flag per visualizzare i dati di debug sul pianificatore locale. Default a False.

        Returns:
            carla.VehicleControl: Il comando di guida per procedere normalmente.
        """
        target_speed = min(
            [self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]
        )
        self._local_planner.set_speed(target_speed)
        control = self._local_planner.run_step(debug=debug)
        return control
    
    def __emergency_stop(self):
        """
        Genera un comando di controllo per l'arresto immediato di emergenza.
        Azzera la propulsione e applica la frenata massima, mantenendo lo sterzo invariato per
        evitare sbandamenti o uscite di corsia, specialmente in curva.

        Args:
            Nessuno.

        Returns:
            carla.VehicleControl: Il comando configurato per la fermata di emergenza.
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
    
    def __update_global_plan(self, overtake_path : list) -> None:
        """
        Aggiorna temporaneamente il piano di navigazione globale per forzare una rotta locale
        specifica, come nel caso di una manovra di elusione o di un sorpasso.

        Args:
            overtake_path (list): Lista di carla.Waypoint che definiscono la traiettoria di sorpasso.

        Returns:
            None
        """
        new_plan = self._local_planner.set_overtake_plan(
            overtake_plan=overtake_path,
            overtake_distance=self._bypass_engine.required_clearance
        )
        self.set_target_speed(2 * self._speed_limit)
        self.set_global_plan(new_plan)
    
    def __update_information(self, debug = False):
        """
        Aggiorna lo stato interno dell'agente raccogliendo i dati ambientali aggiornati.
        Verifica le condizioni meteorologiche (es. pioggia per ridurre la velocità), aggiorna la velocità corrente
        e i limiti, rileva le direzioni previste dai waypoint futuri e decrementa i contatori
        temporali per le manovre tattiche e le soste agli incroci.

        Args:
            debug (bool, opzionale): Flag per abilitare le stampe di debug sul meteo e la telemetria. Default a False.

        Returns:
            None
        """

        self._weather = self._world.get_weather()
        precipitation_intensity = self._weather.precipitation
        preciptitation_deposits = self._weather.precipitation_deposits
        self._is_raining = precipitation_intensity > 50 or preciptitation_deposits > 55

        self._speed = get_speed(self._vehicle)

        self._speed_limit = self._vehicle.get_speed_limit() 
        self._speed_limit -= self._rain_speed_penalty if self._is_raining else 0

        self._local_planner.set_speed(self._speed_limit)

        self._direction = self._local_planner.target_road_option if self._local_planner.target_road_option is not None else RoadOption.LANEFOLLOW

        self._look_ahead_steps = int(self._speed_limit / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps
        )

        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1       

        if self._bypass_engine.evasion_lock > 0:
            self._bypass_engine.evasion_lock -= 1
        else:
            self._bypass_engine.is_bypassing = False

        if self._navigation_engine.is_traversing and self._navigation_engine.active_traversal_frames <= 0:
            self._navigation_engine.is_traversing = False
            self._navigation_engine.active_traversal_frames = 0
            self._navigation_engine.halt_frames = 0
        elif self._navigation_engine.active_traversal_frames > 0:
            self._navigation_engine.active_traversal_frames -= 1

        if self._navigation_engine.starboard_turn_frames > 0:
            self._navigation_engine.starboard_turn_frames -= 1
            
        if self._vehicle.is_at_traffic_light():
            traffic_light = self._vehicle.get_traffic_light()
            if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
                if self._speed > 5.0:
                    print("[INFRACTION] SEMAFORO ROSSO BRUCIATO!")
        
        # Check Rischio Min Speed
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        surrounding_speeds = []
        
        for v in vehicle_list:
            if v.id != self._vehicle.id: 
                if v.get_location().distance(self._vehicle.get_location()) < 50.0:
                    v_speed = v.get_velocity().length() * 3.6
                    if v_speed > 2.0: 
                        surrounding_speeds.append(v_speed)

        # Se ci sono auto in movimento intorno a noi
        if len(surrounding_speeds) > 0:
            avg_traffic_speed = sum(surrounding_speeds) / len(surrounding_speeds)
        
            if avg_traffic_speed > 10.0 and self._speed < (avg_traffic_speed * 0.70):
                if not self._stuck and not self._is_raining and self._direction == RoadOption.LANEFOLLOW:
                    percentage = (self._speed / avg_traffic_speed) * 100
                    print(f"[INFRACTION] Average speed is {percentage:.2f}% of the surrounding traffic's one ({self._speed:.1f} km/h vs Traffico: {avg_traffic_speed:.1f} km/h)")

        if self._is_raining:
            print("[WORLD] It is raining!")
        print("[VEHICLE] Vehicle Speed: {0} - Speed Limit: {1}".format(self._speed, self._speed_limit))

    def _on_collision(self, event):
        """Callback eseguita istantaneamente ogni volta che il veicolo urta qualcosa"""
        actor_type = event.other_actor.type_id
        if "pedestrian" in actor_type:
            print(f"\n[INFRACTION] COLLISIONE CON PEDONE! ID: {event.other_actor.id}")
        elif "vehicle" in actor_type:
            print(f"\n[INFRACTION] COLLISIONE CON VEICOLO! ID: {event.other_actor.id}")
        else:
            print(f"\n[INFRACTION] COLLISIONE CON OSTACOLO! Tipo: {actor_type}")