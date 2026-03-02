# cognitive_modules/signal_evaluator.py
from .base_evaluator import BaseEvaluator


class TrafficSignalEvaluator(BaseEvaluator):
    """Sostituisce lo Scenario 1: Red lights and stops behavior"""

    def evaluate(self, **kwargs):
        sys = self.core_system

        # Invochiamo direttamente il metodo ereditato dal BasicAgent per il controllo semafori
        is_restricted, _ = sys._affected_by_traffic_light(vehicle=sys._vehicle)

        if is_restricted:
            print("[Cognition] -> Halting protocol engaged: Intersection signal restriction detected.")
            return self.halt_vehicle()

        return None