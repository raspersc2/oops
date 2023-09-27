from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from sc2.data import Race
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import ManagerMediator
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    PathUnitToTarget,
    ShootTargetInRange,
    KeepUnitSafe,
)

from ares.cython_extensions.geometry import cy_towards
from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class HighGroundCombat(BaseCombat):
    """When given a high ground target position:
        These units should attempt to target enemies on low ground.

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
    _move_to_high_ground: bool = True

    def retreat_from_high_ground(
        self, enemy_vision_of_high_ground, should_defend: bool
    ) -> bool:
        return self._move_to_high_ground and (
            self.ai.corrected_enemy_race == Race.Zerg
            or should_defend
            or enemy_vision_of_high_ground
        )

    def execute(
        self, units: Union[list[Unit], Units], grid: np.ndarray, **kwargs
    ) -> None:
        attack_target: Point2 = kwargs["attack_target"]
        defend_position: Point2 = kwargs["defend_position"]
        move_to: Point2 = kwargs["move_to"]
        should_defend: bool = kwargs["should_defend"]
        enemy_vision_of_high_ground: bool = self._enemy_on_high_ground(move_to)

        # give up moving to HG if this activates, and take defensive position
        if self.retreat_from_high_ground(enemy_vision_of_high_ground, should_defend):
            self._move_to_high_ground = False

        if not self._move_to_high_ground:
            move_to = cy_towards(defend_position, attack_target, 3.0)

        for unit in units:
            high_ground_maneuver: CombatManeuver = CombatManeuver()

            if self.corrected_time > 90.0 and not self.ai.enemy_units:
                high_ground_maneuver.add(AMove(unit, attack_target))

            elif self._move_to_high_ground:
                high_ground_maneuver.add(self.high_ground_behavior(unit, grid, move_to))
            elif should_defend:
                high_ground_maneuver.add(
                    self.defend_position_behavior(unit, self.ai.race != Race.Zerg)
                )

            else:
                high_ground_maneuver.add(
                    PathUnitToTarget(unit, grid, move_to, success_at_distance=1.0)
                )

            self.ai.register_behavior(high_ground_maneuver)

    def defend_position_behavior(
        self, unit: Unit, target_armoured: bool
    ) -> CombatManeuver:
        protect_pylon: CombatManeuver = CombatManeuver()
        return protect_pylon

    def high_ground_behavior(
        self,
        unit: Unit,
        grid: np.ndarray,
        move_to: Point2,
    ) -> CombatManeuver:
        high_ground_maneuver: CombatManeuver = CombatManeuver()
        if self.ai.enemy_units:
            high_ground_maneuver.add(
                ShootTargetInRange(
                    unit,
                    self.ai.all_enemy_units.filter(
                        lambda u: self.ai.is_visible(u.position)
                    ),
                )
            )
            high_ground_maneuver.add(KeepUnitSafe(unit, grid))
        else:
            high_ground_maneuver.add(PathUnitToTarget(unit, grid, move_to))

        return high_ground_maneuver

    def _enemy_on_high_ground(self, high_ground_spot: Point2) -> bool:
        all_enemy_lower: bool = True
        for enemy in self.ai.enemy_units:
            if self.ai.get_terrain_height(enemy.position) >= self.ai.get_terrain_height(
                high_ground_spot
            ):
                all_enemy_lower = False
                break
        return not all_enemy_lower
