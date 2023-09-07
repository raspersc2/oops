from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import ManagerMediator
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    ShootTargetInRange,
    PathUnitToTarget,
    KeepUnitSafe,
    AMove,
)
from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class HarassPylon(BaseCombat):
    """These units should try to sneak to the enemy pylon.
    But should always attempt to stay safe.

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

    def execute(
        self, units: Union[list[Unit], Units], grid: np.ndarray, **kwargs
    ) -> None:
        harass_position: Point2 = kwargs["harass_position"]
        for unit in units:
            pylon_snipe_maneuver: CombatManeuver = CombatManeuver()

            pylon_snipe_maneuver.add(ShootTargetInRange(unit, self.ai.all_enemy_units))
            pylon_snipe_maneuver.add(
                PathUnitToTarget(unit, grid, harass_position, success_at_distance=3.0)
            )
            if self.corrected_time > 25.0:
                pylon_snipe_maneuver.add(KeepUnitSafe(unit, grid))
            pylon_snipe_maneuver.add(AMove(unit, harass_position))
            self.ai.register_behavior(pylon_snipe_maneuver)
