from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from ares import ManagerMediator
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    AttackTarget,
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    StutterUnitBack,
    StutterUnitForward,
)
from ares.consts import UnitRole, UnitTreeQueryType
from ares.cython_extensions.combat_utils import cy_pick_enemy_target
from ares.cython_extensions.geometry import cy_distance_to, cy_towards
from ares.cython_extensions.units_utils import cy_closest_to
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot

BEST_RANGE: set[UnitTypeId] = {
    UnitTypeId.SIEGETANK,
    UnitTypeId.STALKER,
    UnitTypeId.ROACH,
}

KEEP_ALIVE: set[UnitTypeId] = {UnitTypeId.ZEALOT, UnitTypeId.ZERGLING}

RETREAT_AT_PERC: float = 0.3


@dataclass
class VsGeneric(BaseCombat):
    """Execute behavior for vs zerg, protoss and random

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
    low_health_tags: set[int] = field(default_factory=set)
    attack_target: Point2 = None
    _assigned_units: bool = False
    _calculated_siege_tank_spot: bool = False
    _defend_pylon: bool = False
    _high_ground_behavior: bool = True

    def execute(self, units: Units, grid: np.ndarray, **kwargs) -> None:
        if self.ai.structures and not self._calculated_siege_tank_spot:
            self._calculate_siege_tank_spot()
            self._calculated_siege_tank_spot = True

        self._enemy_in_range_of_pylon()
        self._update_attack_target()
        self._assign_units()
        self._should_do_high_ground_behavior()

        attackers: Units = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        defenders: Units = self.mediator.get_units_from_role(role=UnitRole.DEFENDING)

        for unit in attackers:
            self.ai.register_behavior(
                self._pylon_snipers(unit, grid, self.attack_target)
            )

        for unit in defenders:
            if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                # might want to put targeting behavior here later
                continue
            else:
                self.ai.register_behavior(self._protect_pylon(unit, grid))

    def _assign_units(self) -> None:
        if not self._assigned_units and self.ai.units:
            for i, unit in enumerate(self.ai.units):
                # if (
                #     self.ai.corrected_enemy_race != Race.Zerg
                #     and not self._assigned_units
                #     and unit.type_id not in BEST_RANGE
                # ):
                #     self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
                #     self._assigned_units = True
                # else:
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.DEFENDING)

    def _update_attack_target(self) -> None:
        self.attack_target: Point2 = self.ai.game_info.map_center
        if self.ai.enemy_structures:
            self.attack_target: Point2 = cy_closest_to(
                self.ai.game_info.map_center, self.ai.enemy_structures
            ).position

    def _pylon_snipers(
        self, unit: Unit, grid: np.ndarray, target: Point2
    ) -> CombatManeuver:
        updated_grid: np.ndarray = self._add_influence_to_main_path(grid.copy())
        pylon_snipe_maneuver: CombatManeuver = CombatManeuver()

        pylon_snipe_maneuver.add(ShootTargetInRange(unit, self.ai.all_enemy_units))
        if self.corrected_time > 25.0:
            pylon_snipe_maneuver.add(KeepUnitSafe(unit, updated_grid))
        pylon_snipe_maneuver.add(
            PathUnitToTarget(unit, updated_grid, target, success_at_distance=3.0)
        )
        pylon_snipe_maneuver.add(AMove(unit, target))

        return pylon_snipe_maneuver

    def _protect_pylon(self, unit: Unit, grid: np.ndarray) -> CombatManeuver:
        if self._high_ground_behavior and unit.type_id in BEST_RANGE:
            return self._defensive_ranged_maneuver(unit, grid)

        protect_pylon: CombatManeuver = CombatManeuver()
        # if unit.type_id in KEEP_ALIVE:
        #     if unit.type_id == UnitTypeId.ZEALOT and unit.shield_percentage < 0.2:
        #         self.low_health_tags.add(unit.tag)
        #     elif (
        #         unit.type_id == UnitTypeId.ZERGLING
        #         and unit.health_percentage < RETREAT_AT_PERC
        #     ):
        #         self.low_health_tags.add(unit.tag)
        #
        #     if unit.tag in self.low_health_tags:
        #         if self.ai.race == Race.Protoss and unit.shield_percentage > 0.55:
        #             self.low_health_tags.remove(unit.tag)
        #         if (
        #             self.ai.race == Race.Zerg
        #             and unit.health_percentage > RETREAT_AT_PERC + 0.15
        #         ):
        #             self.low_health_tags.remove(unit.tag)
        #
        #     if unit.tag in self.low_health_tags:
        #         protect_pylon.add(KeepUnitSafe(unit, grid))
        #         return protect_pylon

        if self._defend_pylon:
            in_range: Units = self.ai.all_enemy_units.in_attack_range_of(unit).filter(
                lambda u: self.ai.is_visible(unit.position)
            )
            if len(in_range) > 0 and (unit.ground_range > 2.0):
                target: Unit = cy_pick_enemy_target(in_range)
                if target.is_structure:
                    protect_pylon.add(AMove(unit, target))
                elif self._should_stutter_forward(unit, in_range):
                    protect_pylon.add(StutterUnitForward(unit, target))
                else:
                    protect_pylon.add(StutterUnitBack(unit, target, grid=grid))
            else:
                target: Unit = cy_closest_to(unit.position, self.ai.all_enemy_units)
                protect_pylon.add(AMove(unit, target))
            return protect_pylon
        # not actively defending, but attack things if weapon is ready
        else:
            protect_pylon.add(
                ShootTargetInRange(
                    unit,
                    self.ai.all_enemy_units.filter(
                        lambda u: self.ai.is_visible(u.position)
                    ),
                )
            )

        protect_pylon.add(KeepUnitSafe(unit, grid))
        if self.ai.structures:
            if unit.type_id in BEST_RANGE:
                dist = -4.0
            else:
                dist = -2.0
            move_to: Point2 = Point2(
                cy_towards(
                    self.ai.structures[0].position, self.ai.game_info.map_center, dist
                )
            )

            protect_pylon.add(
                PathUnitToTarget(unit, grid, move_to, success_at_distance=1.4)
            )

        return protect_pylon

    def _defensive_ranged_maneuver(
        self, unit: Unit, grid: np.ndarray
    ) -> CombatManeuver:
        defensive_ranged_maneuver: CombatManeuver = CombatManeuver()

        if self.corrected_time > 90.0 and not self.ai.enemy_units:
            defensive_ranged_maneuver.add(
                AMove(unit, self.ai.enemy_structures[0].position)
            )
            return defensive_ranged_maneuver

        move_to = self._tank_target_spot

        if self._high_ground_behavior:
            defensive_ranged_maneuver.add(
                self.high_ground_behavior(
                    unit, grid, target_armoured=self.ai.race != Race.Zerg
                )
            )

        if self._defend_pylon and self.ai.enemy_units:
            defensive_ranged_maneuver.add(AMove(unit, self.ai.structures[0].position))

        defensive_ranged_maneuver.add(
            PathUnitToTarget(unit, grid, move_to, success_at_distance=1.0)
        )

        return defensive_ranged_maneuver

    def high_ground_behavior(
        self, unit: Unit, grid, target_armoured: bool = False
    ) -> CombatManeuver:
        high_ground_maneuver: CombatManeuver = CombatManeuver()
        if self.ai.enemy_units:
            in_range: Units = self.ai.all_enemy_units.in_attack_range_of(unit)
            in_range = in_range.filter(lambda u: self.ai.is_visible(u.position))
            if len(in_range) > 0:
                target: Unit
                if target_armoured:
                    if armour := in_range.filter(lambda u: u.is_armored):
                        target = cy_pick_enemy_target(armour)
                    else:
                        target: Unit = cy_pick_enemy_target(in_range)
                else:
                    target: Unit = cy_pick_enemy_target(in_range)

                high_ground_maneuver.add(AttackTarget(unit, target))

        return high_ground_maneuver

    def _enemy_on_high_ground(self) -> bool:
        all_enemy_lower: bool = True
        for enemy in self.ai.enemy_units:
            if self.ai.get_terrain_height(enemy.position) >= self.ai.get_terrain_height(
                self._tank_target_spot
            ):
                all_enemy_lower = False
                break
        return not all_enemy_lower

    def _should_stutter_forward(self, unit: Unit, in_range: Units) -> bool:
        return False
        # unit_range: float = unit.ground_range
        # avg_enemy_range: float = sum(u.ground_range for u in in_range) / len(in_range)
        # return unit_range < avg_enemy_range

    def _enemy_in_range_of_pylon(self) -> None:
        if everything_near_pylon := self.mediator.get_units_in_range(
            start_points=self.ai.structures,
            distances=6.5,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=False,
        ):
            near_pylon: Units = everything_near_pylon[0]
            if self.ai.get_total_supply(near_pylon) >= 2:
                self._defend_pylon = True
            return

        self._defend_pylon = False

    def _should_do_high_ground_behavior(self):
        if self._high_ground_behavior and (
            self.ai.corrected_enemy_race == Race.Zerg
            or self._defend_pylon
            or self._enemy_on_high_ground()
        ):
            self._high_ground_behavior = False

    def _add_influence_to_main_path(self, grid: np.ndarray) -> np.ndarray:
        if not self.ai.enemy_structures or self.corrected_time > 45.0:
            return grid

        if path := self.mediator.find_raw_path(
            start=self.ai.structures[0].position,
            target=self.ai.enemy_structures[0].position,
            grid=grid,
            sensitivity=4,
        ):
            map_data = self.mediator.get_map_data_object
            for point in path:
                grid = map_data.add_cost(
                    position=(int(point.x), int(point.y)),
                    radius=4.5,
                    grid=grid,
                    weight=50.0,
                    initial_default_weights=1.0,
                )

        return grid

    def _calculate_siege_tank_spot(self) -> None:
        start_spot: Point2 = self.ai.structures[0].position
        closest_ramp = None
        closest_ramp_dist: float = 999.98
        for ramp in self.mediator.get_map_data_object.map_ramps:
            pos = ramp.top_center
            d = cy_distance_to(start_spot, pos)
            if d < closest_ramp_dist:
                closest_ramp = ramp
                closest_ramp_dist = d

        new_start_spot = self.ai.game_info.map_center
        closest = self.ai.game_info.map_center
        dist = 999.9
        for point in closest_ramp.buildables.points:
            if self.ai.get_terrain_height(point) > self.ai.get_terrain_height(
                start_spot
            ):
                d = cy_distance_to(point, new_start_spot)
                if d < dist:
                    closest = point
                    dist = d

        self._tank_target_spot = closest