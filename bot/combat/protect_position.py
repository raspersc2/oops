from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from sc2.data import Race
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from scipy.interpolate import interp1d

from ares import ManagerMediator
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    StutterUnitBack,
    AMove,
    ShootTargetInRange,
    KeepUnitSafe,
)
from cython_extensions import cy_pick_enemy_target, cy_closest_to
from bot.combat.base_combat import BaseCombat
from bot.consts import BEST_RANGE

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class ProtectPosition(BaseCombat):
    """Protect a provided position. For example, a pylon.
    When not in combat units should take a defensive formation.

    Parameters
    ----------
    ai : AresBot
        Bot object that will be running the game
    config : Dict[Any, Any]
        Dictionary with the data from the configuration file
    mediator : ManagerMediator
        Used for getting information from managers in Ares.
    """

    ai: "AresBot"
    config: dict
    mediator: ManagerMediator
    defensive_concave_positions: dict[int, Point2] = field(default_factory=dict)
    _initial_setup: bool = False

    def execute(
        self, units: Union[list[Unit], Units], grid: np.ndarray, **kwargs
    ) -> None:
        enemy_at_position: Units = kwargs["enemy_at_position"]
        defend_position: Point2 = kwargs["defend_position"]
        should_defend: bool = kwargs["should_defend"]

        if not self._initial_setup and self.ai.structures and self.ai.units:
            self._calculate_defensive_concave(units, defend_position)
            self._initial_setup = True

        for unit in units:
            protect_pylon: CombatManeuver = CombatManeuver()

            if should_defend:
                self.ai.register_behavior(
                    self._defensive_engagement(enemy_at_position, grid, unit)
                )
                continue

            # not actively defending, get in position / shoot things in range
            protect_pylon.add(
                ShootTargetInRange(
                    unit,
                    self.ai.all_enemy_units.filter(
                        lambda u: self.ai.is_visible(u.position)
                    ),
                )
            )
            protect_pylon.add(KeepUnitSafe(unit, grid))
            if unit.tag in self.defensive_concave_positions:
                protect_pylon.add(
                    AMove(unit, self.defensive_concave_positions[unit.tag])
                )
            else:
                protect_pylon.add(AMove(unit, defend_position))
            self.ai.register_behavior(protect_pylon)

    def _defensive_engagement(
        self, enemy_at_pylon: Units, grid: np.ndarray, unit: Unit
    ) -> CombatManeuver:
        defensive_engagement: CombatManeuver = CombatManeuver()
        in_range: Units = self.ai.all_enemy_units.in_attack_range_of(unit).filter(
            lambda u: self.ai.is_visible(unit.position)
        )
        if self.ai.race != Race.Zerg and unit.type_id in BEST_RANGE and enemy_at_pylon:
            if armored := enemy_at_pylon.filter(lambda u: u.is_armored):
                target: Unit = cy_pick_enemy_target(armored)
                defensive_engagement.add(StutterUnitBack(unit, target, grid=grid))
                return defensive_engagement

        if len(in_range) > 0 and (unit.ground_range > 2.0):
            target: Unit = cy_pick_enemy_target(in_range)
            if target.is_structure:
                defensive_engagement.add(AMove(unit, target))
            else:
                defensive_engagement.add(StutterUnitBack(unit, target, grid=grid))
        elif self.ai.all_enemy_units:
            target: Unit = cy_closest_to(unit.position, self.ai.all_enemy_units)
            defensive_engagement.add(AMove(unit, target))
        return defensive_engagement

    def _calculate_defensive_concave(
        self, units: Units, defend_position: Point2
    ) -> None:
        target_location: Point2 = self.ai.enemy_structures[0].position

        setup_from: Point2 = defend_position.position.towards(target_location, 5.5)
        num_points = len(units)
        distance_spread: float = num_points * 0.35
        mid_value_depth = distance_spread * 0.6
        mid_value: float = (
            mid_value_depth if target_location.x < setup_from.x else -mid_value_depth
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

        for pos, unit in zip(points, units):
            self.defensive_concave_positions[unit.tag] = Point2((pos[0], pos[1]))
