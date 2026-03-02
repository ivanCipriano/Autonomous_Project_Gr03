# cognitive_modules/cruise_evaluator.py
from .base_evaluator import BaseEvaluator
from local_planner import RoadOption


class NavigationCruiseEvaluator(BaseEvaluator):
    """Fallback finale: navigazione incroci senza stop e crociera normale"""

    def evaluate(self, **kwargs):
        sys = self.core_system
        debug_mode = kwargs.get('debug', False)

        # Leggiamo la memoria ambientale lasciata da StaticObstructionEvaluator
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