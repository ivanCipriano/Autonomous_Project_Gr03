# cognitive_modules/cruise_evaluator.py
from .base_evaluator import BaseEvaluator
from local_planner import RoadOption


class NavigationCruiseEvaluator(BaseEvaluator):
    """
    Modulo di valutazione di fallback finale responsabile della navigazione in stato di crociera e dell'attraversamento degli incroci.

    Questa classe rappresenta l'ultimo anello della catena decisionale dell'agente autonomo nel simulatore CARLA. 
    Viene interpellata quando nessuna condizione critica (come semafori rossi, ostacoli imminenti o pedoni) 
    è stata rilevata dai valutatori precedenti. Il suo ruolo è garantire una guida fluida e sicura durante la 
    normale marcia e gestire la corretta impostazione della velocità e delle traiettorie durante le svolte agli incroci.
    """

    def evaluate(self, **kwargs):
        """
        Calcola i comandi di controllo per la navigazione standard o per l'immissione negli incroci.

        Il metodo analizza innanzitutto la memoria ambientale (es. la presenza di coni o triangoli di emergenza) 
        lasciata in eredità da valutatori precedenti (come lo StaticObstructionEvaluator). Se l'ego-vehicle è in 
        procinto di attraversare un incrocio con una svolta a destra o a sinistra, la velocità target viene 
        ridotta per garantire una manovra in sicurezza. In assenza di incroci o ostacoli statici, il sistema 
        ripristina l'offset laterale a zero e attiva il comportamento di crociera predefinito, segnalando che 
        il veicolo non è bloccato.

        Args:
            **kwargs: Argomenti variabili. Può includere 'debug' (bool) per attivare la visualizzazione 
                    delle traiettorie e dei dati del planner locale all'interno della simulazione CARLA.

        Returns:
            carla.VehicleControl: Il comando di attuazione contenente i valori di acceleratore, freno 
                                e sterzo necessari per mantenere la crociera o eseguire la svolta.
        """
        sys = self.core_system
        debug_mode = kwargs.get('debug', False)

        hazards = getattr(sys, 'environmental_hazards', {'tw_state': False, 'cone_state': False})
        tw_state = hazards.get('tw_state', False)
        cone_state = hazards.get('cone_state', False)

        if sys._incoming_waypoint.is_junction and (sys._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            print("[Cognition] -> Processing intersection traversal.")
            target_speed = min([sys._behavior.max_speed, sys._speed_limit - 7])
            sys._local_planner.set_speed(target_speed)
            ctrl = sys._local_planner.run_step(debug=debug_mode)

            if sys._navigation_engine.starboard_turn_frames == 0:
                sys._local_planner.set_lateral_offset(0)
            return ctrl

        print("[Cognition] -> Engaging standard navigation cruise.")
        if not tw_state and not cone_state:
            sys._local_planner.set_lateral_offset(0)

        sys._stuck = False
        return sys._BehaviorAgent__normal_behaviour(debug=debug_mode)