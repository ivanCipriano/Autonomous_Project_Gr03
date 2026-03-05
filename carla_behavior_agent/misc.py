#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math
import numpy as np
import carla

def get_speed(vehicle):
    """
    Calcola la velocità scalare di un veicolo convertendola in chilometri orari (km/h).

    Il metodo estrae il vettore velocità tridimensionale dell'attore CARLA e ne calcola la norma euclidea (in m/s),
    moltiplicandola successivamente per 3.6 per ottenere il valore in km/h. Nel contesto della guida autonoma,
    questa metrica è fondamentale per i moduli di controllo longitudinale, per il rispetto dei limiti di velocità
    e per la gestione delle dinamiche di crociera adattiva (ACC).

    Parametri:
        vehicle (carla.Vehicle): L'istanza del veicolo di cui si desidera calcolare la velocità.

    Ritorna:
        float: La velocità corrente del veicolo espressa in km/h.
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def get_acceleration(vehicle):
    """
    Calcola l'accelerazione scalare di un veicolo espressa in metri al secondo quadrato (m/s^2).

    Il metodo recupera il vettore accelerazione tridimensionale restituito dal simulatore CARLA e ne calcola la
    norma euclidea. Nello stack di guida autonoma, questo dato è cruciale per la valutazione del comfort dei
    passeggeri, per il calcolo degli spazi di frenata e per la modellazione della dinamica del veicolo.

    Parametri:
        vehicle (carla.Vehicle): L'istanza del veicolo di cui si desidera calcolare l'accelerazione.

    Ritorna:
        float: L'accelerazione corrente del veicolo espressa in m/s^2.
    """
    acc = vehicle.get_acceleration()

    return math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

def get_trafficlight_trigger_location(traffic_light):
    """
    Calcola la posizione globale esatta del volume di innesco (trigger volume) di un semaforo.

    Questa funzione determina le coordinate del punto che rappresenta l'area di influenza del semaforo, applicando
    trasformazioni spaziali. Ruota il vettore che rappresenta l'estensione del volume di innesco in base
    all'angolo di imbardata (yaw) del semaforo e lo somma alla posizione globale dello stesso. Questo è essenziale
    nei sistemi di guida autonoma per individuare la linea di arresto esatta agli incroci regolati da semaforo.

    Parametri:
        traffic_light (carla.TrafficLight): L'istanza del semaforo di cui calcolare l'area di innesco.

    Ritorna:
        carla.Location: Le coordinate tridimensionali globali del trigger volume del semaforo.
    """
    def rotate_point(point, radians):
        """
        Funzione interna di supporto per ruotare un punto 2D di un angolo specifico.

        Applica la matrice di rotazione bidimensionale standard (sugli assi X e Y) a un punto vettoriale,
        mantenendo inalterata la coordinata Z. È utilizzata per calcolare gli offset spaziali orientati.

        Parametri:
            point (carla.Vector3D): Il punto vettoriale da ruotare.
            radians (float): L'angolo di rotazione espresso in radianti.

        Ritorna:
            carla.Vector3D: Il nuovo vettore risultante dalla rotazione.
        """
        rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
        rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

        return carla.Vector3D(rotated_x, rotated_y, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)


def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Verifica se un target si trova entro un raggio specifico e in un determinato intervallo angolare rispetto a un riferimento.

    Il metodo calcola innanzitutto la distanza euclidea tra due trasformazioni spaziali. Se il target rientra nella
    distanza massima, verifica opzionalmente (se fornito) che si trovi all'interno di un campo visivo angolare specifico,
    calcolando il prodotto scalare tra il vettore direzione e il vettore frontale del riferimento. Questo filtro
    è di primaria importanza nella pipeline di percezione per scartare ostacoli irrilevanti (es. veicoli alle spalle).

    Parametri:
        target_transform (carla.Transform): La trasformazione (posizione e orientamento) dell'oggetto target.
        reference_transform (carla.Transform): La trasformazione (posizione e orientamento) dell'oggetto di riferimento.
        max_distance (float): Distanza massima consentita in metri.
        angle_interval (list, opzionale): Intervallo angolare [min, max] in gradi (0 indica frontale, 180 posteriore).

    Ritorna:
        bool: True se il target rispetta i vincoli di distanza e angolazione, False altrimenti.
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    if not angle_interval:
        return True

    min_angle = angle_interval[0]
    max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return min_angle < angle < max_angle

def vector(location_1, location_2):
    """
    Calcola il vettore direzionale unitario (normalizzato) tra due punti spaziali 3D.

    Estrae le differenze tra le componenti X, Y e Z delle due posizioni e divide ciascuna componente per la norma
    euclidea del vettore risultante. Nel contesto della guida autonoma, questo è essenziale per generare le
    direttrici di traiettoria e calcolare gli angoli di sterzata necessari al local planner.

    Parametri:
        location_1 (carla.Location): Le coordinate del punto di origine.
        location_2 (carla.Location): Le coordinate del punto di destinazione.

    Ritorna:
        list: Una lista di 3 float [x, y, z] rappresentante il vettore direzione normalizzato.
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

def compute_distance(location_1, location_2):
    """
    Calcola la distanza euclidea standard tra due punti 3D nello spazio della simulazione.

    Utilizza la norma lineare (np.linalg.norm) sulle differenze delle coordinate cartesiane. Introduce un piccolissimo
    valore epsilon (np.finfo(float).eps) per evitare divisioni per zero in computazioni a valle.

    Parametri:
        location_1 (carla.Location): Il primo punto tridimensionale.
        location_2 (carla.Location): Il secondo punto tridimensionale.

    Ritorna:
        float: La distanza euclidea tra i due punti espressa in metri.
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def compute_distance_from_center(actor1, actor2 = None, distance = 5):
    """
    Calcola la distanza effettiva tra i bordi (bounding boxes) di due attori partendo dalla distanza dei loro centri.

    Nei sistemi di guida autonoma, la distanza centro-centro può portare a collisioni se i veicoli sono lunghi
    (es. autobus). Questo metodo sottrae la dimensione massima delle bounding box dei veicoli sull'asse X o Y
    dalla distanza totale fornita, restituendo lo spazio libero effettivo (distanza bordo-bordo).

    Parametri:
        actor1 (carla.Actor): Il primo attore CARLA coinvolto.
        actor2 (carla.Actor, opzionale): Il secondo attore CARLA. Se non fornito, viene considerato nullo.
        distance (float, opzionale): La distanza iniziale misurata tra i centri degli attori. Default a 5 metri.

    Ritorna:
        float: La distanza effettiva al netto dell'ingombro fisico dei veicoli.
    """
    actor1_extent = max(actor1.bounding_box.extent.x, actor1.bounding_box.extent.y)
    actor2_extent = max(actor2.bounding_box.extent.x, actor2.bounding_box.extent.y) if actor2 else 0
    return distance - actor1_extent - actor2_extent

def get_distance(a, b):
    """
    Calcola la distanza euclidea tra due oggetti CARLA generici, indipendentemente dal loro tipo specifico.

    Questa funzione funge da wrapper polimorfico che estrae la coordinata spaziale (carla.Location) da entità
    differenti come Actor, Landmark, Waypoint o Transform. Uniformare il calcolo della distanza è vitale per
    i moduli di decisione che devono confrontare simultaneamente elementi eterogenei (es. distanza tra
    un veicolo e un semaforo, oppure tra un waypoint e un pedone).

    Parametri:
        a (variabile): Il primo oggetto (carla.Actor, carla.Landmark, carla.Waypoint, carla.Transform, carla.Location).
        b (variabile): Il secondo oggetto (stessi tipi ammessi).

    Ritorna:
        float: La distanza euclidea misurata in metri.
    """

    def extract_location(obj):
        """
        Funzione interna di supporto per isolare l'oggetto carla.Location da vari tipi di strutture dati CARLA.

        Attraverso controlli sul tipo di dato (isinstance), naviga le proprietà specifiche dell'oggetto per estrarre
        le sue coordinate pure. Solleva un'eccezione se il tipo di dato non è supportato.

        Parametri:
            obj (variabile): L'oggetto di origine da cui estrarre le coordinate.

        Ritorna:
            carla.Location: L'oggetto CARLA contenente le coordinate X, Y, Z.

        Solleva:
            ValueError: Se il tipo dell'oggetto fornito non rientra tra quelli supportati dall'API per l'estrazione.
        """
        if isinstance(obj, carla.Location):
            return obj
        elif isinstance(obj, carla.Transform):
            return obj.location
        elif isinstance(obj, carla.Waypoint):
            return obj.transform.location
        elif isinstance(obj, carla.Landmark):
            return obj.waypoint.transform.location
        elif isinstance(obj, carla.Actor):
            return obj.get_location()
        else:
            raise ValueError(f"Invalid input type: {type(obj).__name__}. "
                             "Expected carla.Actor, carla.Landmark, carla.Transform, carla.Waypoint, or carla.Location.")

    loc_a = extract_location(a)
    loc_b = extract_location(b)

    return loc_a.distance(loc_b)