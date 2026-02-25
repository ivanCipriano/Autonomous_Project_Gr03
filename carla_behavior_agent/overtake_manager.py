import carla
import math
from local_planner import RoadOption


class OvertakeManager:
    def __init__(self, vehicle, world, carla_map, global_route_planner):
        self._vehicle = vehicle
        self._world = world
        self._map = carla_map
        self._global_route_planner = global_route_planner
        self.in_overtake = False

    def check_opposite_vehicle(self, ego_wp, search_distance):
        """
        Controlla se ci sono veicoli in arrivo dalla corsia opposta entro una certa distanza,
        valutando SOLO i veicoli che si trovano DAVANTI all'ego-vehicle.
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        # Estraiamo posizione e direzione verso cui guarda la nostra auto
        ego_transform = self._vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()

        for v in vehicle_list:
            if v.id == self._vehicle.id:
                continue

            v_loc = v.get_location()
            v_wp = self._map.get_waypoint(v_loc)

            # Verifica se il veicolo si trova nella corsia opposta
            if v_wp.lane_id == ego_wp.lane_id * -1:

                # Calcoliamo il vettore di direzione verso il veicolo target (ignoriamo l'asse Z)
                dir_vector = carla.Location(v_loc.x - ego_loc.x, v_loc.y - ego_loc.y, 0.0)

                # Prodotto scalare tra la nostra direzione frontale e la posizione dell'altra auto
                dot_product = (ego_forward.x * dir_vector.x) + (ego_forward.y * dir_vector.y)

                # Se il prodotto scalare è POSITIVO, l'auto si trova nel nostro emicampo VISIVO FRONTALE
                if dot_product > 0:
                    dist_to_v = ego_loc.distance(v_loc)
                    if dist_to_v < search_distance:
                        return True  # Arriva qualcuno davanti a noi!

        return False  # Nessuno davanti a noi: via libera!

    def get_overtake_path(self, ego_wp, obstacle_distance, end_destination_loc):
        """
        Genera la rotta di sorpasso proiettando i waypoint in avanti sulla corsia corrente
        e spostandoli a sinistra. Questo risolve il bug della corsia opposta in CARLA.
        """
        safe_search_distance = obstacle_distance + 70.0
        is_traffic_incoming = self.check_opposite_vehicle(ego_wp, safe_search_distance)

        if is_traffic_incoming:
            return None  # Arriva qualcuno, abortire manovra e aspettare

        overtake_plan = []

        # Punto 1: Immissione diagonale nella corsia di sinistra (a 10 metri avanti a noi)
        forward_wp1 = ego_wp.next(10.0)
        if not forward_wp1: return None
        wp1_left = forward_wp1[0].get_left_lane()

        if not wp1_left or wp1_left.lane_type != carla.LaneType.Driving:
            return None  # Impossibile sorpassare, linea continua o niente corsia

        overtake_plan.append((wp1_left, RoadOption.CHANGELANELEFT))

        # Punto 2: Superamento dell'ostacolo.
        # Avanziamo nella nostra corsia fino a superare l'ostacolo di 10 metri, e prendiamo la proiezione a sinistra
        forward_wp2 = ego_wp.next(obstacle_distance + 10.0)
        if forward_wp2:
            wp2_left = forward_wp2[0].get_left_lane()
            if wp2_left:
                overtake_plan.append((wp2_left, RoadOption.LANEFOLLOW))

        # Punto 3: Rientro obliquo nella nostra corsia originale
        # Avanziamo di 25 metri oltre l'ostacolo e usiamo il waypoint della nostra corsia (senza spostarlo a sx)
        forward_wp3 = ego_wp.next(obstacle_distance + 25.0)
        if not forward_wp3: return None
        wp3_right = forward_wp3[0]
        overtake_plan.append((wp3_right, RoadOption.CHANGELANERIGHT))

        # 4. Chiediamo al Global Planner di ricalcolare il percorso fino all'arrivo
        # partendo dal punto esatto in cui siamo rientrati in corsia
        rest_of_route = self._global_route_planner.trace_route(wp3_right.transform.location, end_destination_loc)

        # Uniamo la manovra fluida con il resto del viaggio e la restituiamo al LocalPlanner
        return overtake_plan + rest_of_route