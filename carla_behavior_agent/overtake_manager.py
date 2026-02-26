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
        self.original_lane_id = None  # Nuova variabile per memorizzare la corsia di partenza

    def check_opposite_vehicle(self, ego_wp, search_distance):
        """
        Controlla se ci sono veicoli in arrivo dalla corsia opposta entro una certa distanza,
        valutando SOLO i veicoli che si trovano DAVANTI all'ego-vehicle.
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        ego_transform = self._vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()

        for v in vehicle_list:
            if v.id == self._vehicle.id:
                continue

            v_loc = v.get_location()
            v_wp = self._map.get_waypoint(v_loc)

            if v_wp.lane_id == ego_wp.lane_id * -1:
                dir_vector = carla.Location(v_loc.x - ego_loc.x, v_loc.y - ego_loc.y, 0.0)
                dot_product = (ego_forward.x * dir_vector.x) + (ego_forward.y * dir_vector.y)

                if dot_product > 0:
                    dist_to_v = ego_loc.distance(v_loc)
                    if dist_to_v < search_distance:
                        return True
        return False

    def get_overtake_path(self, ego_wp, obstacle_distance, end_destination_loc):
        """
        Genera la rotta di sorpasso e ottimizza il rientro dopo l'ultimo cono.
        """
        safe_search_distance = obstacle_distance + 60.0
        is_traffic_incoming = self.check_opposite_vehicle(ego_wp, safe_search_distance)

        if is_traffic_incoming:
            return None

            # Salviamo l'ID della corsia originale per capire in futuro quando il rientro è completo
        self.original_lane_id = ego_wp.lane_id

        overtake_plan = []

        # 1. Immissione dolce nella corsia di sinistra
        forward_wp1 = ego_wp.next(10.0)
        if not forward_wp1: return None
        wp1_left = forward_wp1[0].get_left_lane()

        if not wp1_left or wp1_left.lane_type != carla.LaneType.Driving:
            return None

        overtake_plan.append((wp1_left, RoadOption.CHANGELANELEFT))

        # --- LOGICA 5: Controllo Coni per Ritardo Rientro (MIGLIORATA) ---
        # Rientro stretto: base 8 metri oltre l'ostacolo (anziché 15)
        max_pass_distance = obstacle_distance + 8.0

        cones = self._world.get_actors().filter("*static.prop.constructioncone*")
        for cone in cones:
            cone_wp = self._map.get_waypoint(cone.get_location())

            # Controlliamo i coni solo nella nostra corsia originale
            if cone_wp.lane_id == ego_wp.lane_id:
                dist_to_cone = ego_wp.transform.location.distance(cone.get_location())

                # Se fa parte del cantiere, estendiamo il sorpasso ma solo 4 metri oltre l'ultimo cono
                if obstacle_distance - 5.0 < dist_to_cone < (max_pass_distance + 15.0):
                    new_pass_distance = dist_to_cone + 4.0
                    if new_pass_distance > max_pass_distance:
                        max_pass_distance = new_pass_distance
                        print(
                            f"[OvertakeManager] Coni in rientro rilevati. Rientro ravvicinato a {max_pass_distance:.1f}m!")
        # ---------------------------------------------------

        # 2. Superamento
        forward_wp2 = ego_wp.next(max_pass_distance)
        if forward_wp2:
            wp2_left = forward_wp2[0].get_left_lane()
            if wp2_left:
                overtake_plan.append((wp2_left, RoadOption.LANEFOLLOW))

        # 3. Rientro obliquo nella corsia originale (taglio più rapido, +10m anziché +15m)
        forward_wp3 = ego_wp.next(max_pass_distance + 10.0)
        if not forward_wp3: return None
        wp3_right = forward_wp3[0]
        overtake_plan.append((wp3_right, RoadOption.CHANGELANERIGHT))

        # 4. Ricalcolo
        rest_of_route = self._global_route_planner.trace_route(wp3_right.transform.location, end_destination_loc)

        return overtake_plan + rest_of_route