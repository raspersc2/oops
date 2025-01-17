import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import UseAbility
from ares.behaviors.combat.individual.siege_tank_decision import SiegeTankDecision
from ares.dicts.unit_data import UNIT_DATA
from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_center, cy_towards
from loguru import logger
from s2clientprotocol.data_pb2 import AbilityData
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from scipy.interpolate import interp1d

from bot.combat_squads.squad.base_squad import BaseSquad
from bot.combat_squads.squad.feed_back import FeedBack


@dataclass
class SquadSetup(BaseSquad):
    ai: "AresBot"
    mediator: ManagerMediator
    squad: UnitSquad
    target: Point2

    core_concave_positions: dict[int, Point2] = field(default_factory=dict)
    fodder_concave_positions: dict[int, Point2] = field(default_factory=dict)

    def __post_init__(self):
        units: list[Unit] = self.squad.squad_units
        fodder_tags: set[int] = self.get_fodder_tags(units)

        if non_fodder := [
            u
            for u in units
            if u.tag not in fodder_tags and not UNIT_DATA[u.type_id]["flying"]
        ]:
            non_fodder_move_to: Point2 = Point2(
                cy_towards(self.squad.squad_position, self.target, 0.5)
            )

            self.core_concave_positions: dict[int, Point2] = self._calculate_concave(
                non_fodder, non_fodder_move_to, self.target
            )

        if fodder := [
            u
            for u in units
            if u.tag in fodder_tags and not UNIT_DATA[u.type_id]["flying"]
        ]:
            fodder_move_to: Point2 = Point2(
                cy_towards(self.squad.squad_position, self.target, 2.2)
            )

            self.fodder_concave_positions: dict[int, Point2] = self._calculate_concave(
                fodder, fodder_move_to, self.target
            )

    def execute(
        self,
        squad: UnitSquad,
        enemy: Union[Units, list[Unit]],
        target: Point2,
        **kwargs,
    ) -> None:
        units: list[Unit] = squad.squad_units

        for unit in units:
            if UNIT_DATA[unit.type_id]["flying"]:
                unit.move(squad.squad_position)
                continue
            fodder_maneuver: CombatManeuver = CombatManeuver()
            # fodder_maneuver.add(SiegeTankDecision(unit, enemy, target))
            fodder_maneuver.add(FeedBack(unit, enemy))
            tag: int = unit.tag
            if tag in self.core_concave_positions:
                pos: Point2 = self.core_concave_positions[tag]
                if self.ai.in_pathing_grid(pos):
                    fodder_maneuver.add(UseAbility(AbilityId.MOVE_MOVE, unit, pos))
            elif tag in self.fodder_concave_positions:
                pos: Point2 = self.fodder_concave_positions[tag]
                if self.ai.in_pathing_grid(pos):
                    fodder_maneuver.add(UseAbility(AbilityId.MOVE_MOVE, unit, pos))
            self.ai.register_behavior(fodder_maneuver)

    def _calculate_concave(
        self,
        units: list[Unit],
        defend_position: Point2,
        target_location: Point2,
        setup_towards: float = 0.2,
    ) -> dict[int, Point2]:
        setup_from: Point2 = Point2(
            cy_towards(defend_position, target_location, setup_towards)
        )
        num_points = len(units)
        distance_spread: float = num_points * 0.35
        mid_value_depth = distance_spread * 0.6
        mid_value: float = (
            mid_value_depth if target_location[0] < setup_from[0] else -mid_value_depth
        )
        point_a: Point2 = Point2((setup_from[0], setup_from[1] - distance_spread))
        mid_point: Point2 = Point2((setup_from[0] + mid_value, setup_from[1]))
        point_b: Point2 = Point2((setup_from[0], setup_from[1] + distance_spread))

        points = np.array(
            [[point_a.x, mid_point.x, point_b.x], [point_a.y, mid_point.y, point_b.y]]
        ).T

        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]

        alpha = np.linspace(0, 1, num_points)
        interpolator = interp1d(distance, points, kind="quadratic", axis=0)
        points = interpolator(alpha)

        concave_positions: dict[int, Point2] = dict()
        for pos, unit in zip(points, units):
            concave_positions[unit.tag] = Point2((pos[0], pos[1]))
        return concave_positions

    # alternative working solution
    # def _calculate_concave(
    #     self,
    #     units: list[Unit],
    #     defend_position: Point2,
    #     target_location: Point2,
    #     arc_radius: float = 5.0,  # Adjustable radius of the concave
    #     arc_angle: float = math.pi / 2,  # Angle in radians (default: 90° concave)
    # ) -> dict[int, Point2]:
    #     if not units:
    #         return {}
    #
    #     # Calculate the arc center slightly offset from defend_position towards target_location
    #     center_offset = defend_position.towards(target_location, arc_radius * 0.5)
    #
    #     # Number of units determines arc segment distribution
    #     num_units = len(units)
    #     angle_step = arc_angle / (num_units - 1) if num_units > 1 else 0
    #     start_angle = -arc_angle / 2  # Start at the leftmost edge of the arc
    #
    #     # Calculate positions for units along the arc
    #     concave_positions = {}
    #     for i, unit in enumerate(units):
    #         angle = start_angle + i * angle_step
    #         x = center_offset.x + arc_radius * math.cos(angle)
    #         y = center_offset.y + arc_radius * math.sin(angle)
    #         concave_positions[unit.tag] = Point2((x, y))
    #
    #     return concave_positions
