from .base_evaluator import BaseEvaluator


class TrafficSignalEvaluator(BaseEvaluator):
    """
    Valuta lo stato dei segnali semaforici per determinare le azioni del veicolo.

    Questa classe estende BaseEvaluator per gestire specificamente la logica
    legata agli incroci regolati da semafori, imponendo l'arresto del veicolo
    quando viene rilevato un segnale di stop (es. semaforo rosso).
    """

    def evaluate(self, **kwargs):
        """
        Analizza l'ambiente circostante per verificare la presenza di restrizioni semaforiche.

        Interroga il sistema centrale per determinare se il veicolo è attualmente
        soggetto a un semaforo restrittivo. In caso affermativo, blocca l'avanzamento
        e attiva il protocollo di arresto.

        Args:
            **kwargs: Argomenti chiave arbitrari. Mantenuti per compatibilità con la
                      firma del metodo della classe base `BaseEvaluator`.

        Returns:
            object o None: Restituisce l'output di un comando di arresto se il veicolo deve fermarsi al semaforo.
                           Restituisce `None` se non ci sono restrizioni semaforiche.
        """
        sys = self.core_system

        is_restricted, _ = sys._affected_by_traffic_light(vehicle=sys._vehicle)

        if is_restricted:
            print("[Cognition] -> Halting protocol engaged: Intersection signal restriction detected.")
            return self.halt_vehicle()

        return None