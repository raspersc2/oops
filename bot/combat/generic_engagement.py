from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2

from ares import ManagerMediator
from ares.behaviors.combat.individual import CombatIndividualBehavior, KeepUnitSafe
from ares.consts import ALL_STRUCTURES
from ares.behaviors.combat.individual import (
    AMove,
    StutterUnitBack,
    StutterUnitForward,
    UseAbility,
)
from cython_extensions import cy_closest_to, cy_distance_to
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

            if (
                unit.type_id != UnitID.BANELING
                and unit.is_light
                and [
                    e
                    for e in self.nearby_targets
                    if e.type_id == UnitID.BANELING
                    and cy_distance_to(unit.position, e.position) < 3.8
                ]
            ):
                return KeepUnitSafe(unit, grid).execute(ai, config, mediator)
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
                    # low health, but we are faster and have more range
                    # always stay out of danger where possible
                    enemy_speed: float = enemy_target.movement_speed
                    if ai.enemy_race == Race.Zerg and ai.has_creep(
                        enemy_target.position
                    ):
                        enemy_speed *= 1.3
                    own_speed: float = unit.movement_speed
                    if ai.race == Race.Zerg and ai.has_creep(unit.position):
                        own_speed *= 1.3
                    if (
                        not enemy_target.is_flying
                        and own_speed > enemy_speed
                        and unit.ground_range > enemy_target.ground_range
                        and unit.shield_health_percentage < 0.25
                        and cy_distance_to(unit.position, enemy_target.position)
                        < enemy_target.ground_range + enemy_target.radius + unit.radius
                    ):
                        return KeepUnitSafe(unit, grid).execute(ai, config, mediator)
                    else:
                        return StutterUnitBack(
                            unit=unit, target=enemy_target, grid=grid
                        ).execute(ai, config, mediator)
