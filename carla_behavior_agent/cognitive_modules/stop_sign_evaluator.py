from .base_evaluator import BaseEvaluator
from misc import compute_distance_from_center, get_distance


class StopEvaluator(BaseEvaluator):
    """
    Valuta e gestisce il comportamento del veicolo in presenza di segnali di STOP obbligatori.

    Questa classe estende BaseEvaluator per identificare i segnali di STOP ("206")
    lungo il percorso, calcolare la distanza da essi e determinare l'azione corretta:
    arrestare il veicolo all'incrocio, accodarsi al veicolo che precede tramite
    cruise control adattivo, o rallentare in fase di approccio.
    """

    def _detect_stop_regulation(self, vehicle, sign_distance=20):
        sys = self.core_system
        is_affected, signal_obj = sys._affected_by_sign(vehicle=vehicle, sign_type="206", max_distance=sign_distance)
        distance = -1 if not is_affected else get_distance(a=vehicle, b=signal_obj)
        return is_affected, signal_obj, distance

    def evaluate(self, **kwargs):
        """
        Rileva la presenza di un segnale di STOP di pertinenza per il veicolo.

        Interroga il sistema centrale per verificare se c'è un cartello di tipo "206"
        (STOP) entro una certa distanza che influenza la traiettoria corrente.

        Args:
            vehicle (object): L'oggetto che rappresenta l'ego vehicle.
            sign_distance (int, opzionale): Distanza massima di scansione in metri.
                                            Il valore di default è 20.

        Returns:
            tuple: Una tupla di tre elementi contenente:
                - bool: True se è stato rilevato un segnale di STOP valido, False altrimenti.
                - object o None: L'oggetto del segnale rilevato, None se assente.
                - float: La distanza dal segnale in metri, oppure -1 se assente.
        """
        sys = self.core_system
        current_wp = kwargs.get('ego_vehicle_wp')

        v_state, lead_v, _ = self.scan_for_fleet(current_wp)
        is_stop, _, s_dist = self._detect_stop_regulation(vehicle=sys._vehicle, sign_distance=20)

        if is_stop:
            actual_dist = compute_distance_from_center(actor1=sys._vehicle, distance=s_dist)
            print(f"[Cognition] -> Intersection regulation (STOP) registered at {actual_dist:.1f}m.")

            if actual_dist < 2 and not sys._navigation_engine.is_traversing:
                print("--- [Cognition] Executing mandatory intersection halt.")
                ctrl = sys._navigation_engine.evaluate_traversal_safety(sys._local_planner, preview_steps=sys._look_ahead_steps)
                if ctrl:
                    sys.set_target_speed(sys._speed_limit)
                    return sys._navigation_engine.execute_traversal(routing_directive=sys._direction)

                return self.halt_vehicle()
            elif v_state and actual_dist > 5:
                print("--- [Cognition] Stop distant. Switching to adaptive cruise for lead vehicle.")
                follow_dist = compute_distance_from_center(actor1=sys._vehicle, actor2=lead_v,
                                                           distance=get_distance(sys._vehicle, lead_v))
                return self.adaptive_cruise_control(target_vehicle=lead_v, distance=follow_dist)
            else:
                print("--- [Cognition] Stop distant. Decelerating to approach velocity.")
                sys.set_target_speed(sys._approach_speed)
                return sys._local_planner.run_step()

        return None