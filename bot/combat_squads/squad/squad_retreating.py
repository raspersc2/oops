from dataclasses import dataclass
from typing import Union

import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    ShootTargetInRange,
    UseAbility,
)
from ares.dicts.unit_data import UNIT_DATA
from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from cython_extensions.combat_utils import cy_pick_enemy_target
from cython_extensions.geometry import cy_distance_to_squared, cy_towards
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat_squads.squad.base_squad import BaseSquad


@dataclass
class SquadRetreating(BaseSquad):
    ai: "AresBot"
    mediator: ManagerMediator
    squad: UnitSquad
    target: Point2

    def execute(
        self,
        squad: UnitSquad,
        enemy: Union[Units, list[Unit]],
        target: Point2,
        **kwargs,
    ) -> None:
        grid: np.ndarray = self.mediator.get_ground_grid
        units: list[Unit] = squad.squad_units
        retreat_position: Point2
        rough_retreat_spot: Point2 = Point2(
            cy_towards(squad.squad_position, target, -12.5)
        )
        if squad.main_squad and self.ai.in_pathing_grid(rough_retreat_spot):
            retreat_position = self.mediator.find_closest_safe_spot(
                from_pos=rough_retreat_spot,
                grid=grid,
            )
        elif "pos_of_main_squad" in kwargs:
            retreat_position = kwargs["pos_of_main_squad"]
        else:
            retreat_position = target

        for unit in units:
            # close melee should carry on fighting
            if (
                unit.ground_range < 3
                and not UNIT_DATA[unit.type_id]["flying"]
                and unit.can_attack
                and enemy
                and (
                    close_ground := [
                        e
                        for e in enemy
                        if not UNIT_DATA[e.type_id]["flying"]
                        and cy_distance_to_squared(e.position, unit.position) < 10.0
                    ]
                )
            ):
                unit.attack(cy_pick_enemy_target(close_ground))
                continue

            retreat_maneuver: CombatManeuver = CombatManeuver()
            retreat_maneuver = self._use_unit_abilities(
                unit, enemy, grid, squad, target, retreat_maneuver
            )
            retreat_maneuver.add(ShootTargetInRange(unit, enemy))
            retreat_maneuver.add(KeepUnitSafe(unit, grid))
            retreat_maneuver.add(
                UseAbility(AbilityId.MOVE_MOVE, unit, retreat_position)
            )

            self.ai.register_behavior(retreat_maneuver)
