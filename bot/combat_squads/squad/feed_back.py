from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from ares.behaviors.combat.individual.combat_individual_behavior import (
    CombatIndividualBehavior,
)
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions.geometry import cy_distance_to, cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit
from sc2.units import Units

if TYPE_CHECKING:
    from ares import AresBot

FEED_BACK_RANGE: float = 10.0


@dataclass
class FeedBack(CombatIndividualBehavior):
    """A-Move a unit to a target.

    Example:
    ```py
    from ares.behaviors.combat.individual import AMove

    self.register_behavior(AMove(unit, self.game_info.map_center))
    ```

    Attributes:
        unit: The unit to stay safe.
        target: Where the unit is going.

    """

    unit: Unit
    targets: Union[list[Unit], Units]
    extra_range: float = 0.0

    def execute(self, ai: "AresBot", config: dict, mediator: ManagerMediator) -> bool:
        if AbilityId.FEEDBACK_FEEDBACK not in self.unit.abilities:
            return False

        if targets := [
            t
            for t in self.targets
            if t.energy >= 50
            and cy_distance_to(t.position, self.unit.position)
            < FEED_BACK_RANGE + t.radius + self.unit.radius + self.extra_range
        ]:
            target_with_most_energy: Unit = max(targets, key=lambda t: t.energy)
            self.unit(AbilityId.FEEDBACK_FEEDBACK, target_with_most_energy)
            return True
        return False
