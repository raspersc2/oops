from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2

from ares import ManagerMediator
from ares.behaviors.combat.individual import CombatIndividualBehavior
from ares.consts import ALL_STRUCTURES
from ares.behaviors.combat.individual import (
    AMove,
    StutterUnitBack,
    StutterUnitForward,
    UseAbility,
)
from cython_extensions import cy_closest_to
from sc2.unit import Unit
from sc2.units import Units

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class GenericEngagement(CombatIndividualBehavior):
    """A very opinionated behavior to control any melee or ranged unit
    in battle.

    Attributes
    ----------
    unit: Unit
        The unit we want to control.
    nearby_targets : Units
        The nearby enemy we want to fight.
    stutter_forward : bool
        Should unit aggressively fight?
        If set to `False` unit will kite back.
    use_blink : bool
        Use blink if available?
        default = True
    """

    unit: Unit
    nearby_targets: Union[Units, list[Unit]]
    stutter_forward: bool
    use_blink: bool = True

    def execute(
        self, ai: "AresBot", config: dict, mediator: ManagerMediator, **kwargs
    ) -> bool:
        """Shoot at the target if possible, else kite back.

        Parameters
        ----------
        ai : AresBot
            Bot object that will be running the game
        config :
            Dictionary with the data from the configuration file
        mediator :
            ManagerMediator used for getting information from other managers.
        **kwargs :
            None

        Returns
        -------
        bool :
            CombatBehavior carried out an action.
        """

        unit = self.unit
        nearby_targets = self.nearby_targets
        if non_structures := [
            u for u in nearby_targets if u.type_id not in ALL_STRUCTURES
        ]:
            if tanks := [
                u for u in non_structures if u.type_id == UnitID.SIEGETANKSIEGED
            ]:
                enemy_target: Unit = cy_closest_to(unit.position, tanks)
            else:
                enemy_target: Unit = cy_closest_to(unit.position, non_structures)
        else:
            enemy_target: Unit = cy_closest_to(unit.position, nearby_targets)
        if self.stutter_forward:
            if (
                self.use_blink
                and AbilityId.EFFECT_BLINK_STALKER in unit.abilities
                and ai.is_visible(enemy_target.position)
            ):
                return UseAbility(
                    ability=AbilityId.EFFECT_BLINK_STALKER,
                    unit=unit,
                    target=enemy_target.position,
                ).execute(ai, config, mediator)
            else:
                return StutterUnitForward(unit, enemy_target).execute(
                    ai, config, mediator
                )
        else:
            grid: np.ndarray = mediator.get_ground_grid
            if unit.is_flying:
                grid = mediator.get_air_grid
            if self.unit.ground_range < 3.0:
                return AMove(unit=unit, target=enemy_target).execute(
                    ai, config, mediator
                )
            else:
                if (
                    unit.shield_percentage < 0.2
                    and AbilityId.EFFECT_BLINK_STALKER in unit.abilities
                ):
                    safe_spot: Point2 = mediator.find_closest_safe_spot(
                        from_pos=unit.position, grid=grid
                    )
                    return UseAbility(
                        ability=AbilityId.EFFECT_BLINK_STALKER,
                        unit=unit,
                        target=safe_spot,
                    ).execute(ai, config, mediator)
                else:
                    return StutterUnitBack(
                        unit=unit, target=enemy_target, grid=grid
                    ).execute(ai, config, mediator)
