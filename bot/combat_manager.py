from typing import TYPE_CHECKING, Optional

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.data import Race
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import ManagerMediator
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import StutterUnitBack, UseAbility
from ares.consts import UnitRole, UnitTreeQueryType
from cython_extensions import cy_distance_to, cy_closest_to

from bot.combat.base_combat import BaseCombat
from bot.combat.generic_engagement import GenericEngagement
from bot.consts import BEST_RANGE
from bot.combat.harass_pylon import HarassPylon
from bot.combat.high_ground_combat import HighGroundCombat
from bot.combat.protect_position import ProtectPosition

if TYPE_CHECKING:
    from ares import AresBot


class CombatManager:
    MELEE_FLEE_AT_PERC: float = 0.3

    """
    Store combat related state.
    And orchestrate the combat classes

        Parameters
    ----------
    ai : AresBot
        Bot object that will be running the game
    config : Dict[Any, Any]
        Dictionary with the data from the configuration file
    mediator : ManagerMediator
        Used for getting information from managers in Ares.
    """

    def __init__(self, ai: "AresBot", config: dict, mediator: ManagerMediator):
        self.ai: "AresBot" = ai
        self.config: dict = config
        self.mediator: ManagerMediator = mediator
        self._assigned_units: bool = False
        self._close_high_ground_spots: list[Point2] = []
        self._defend_pylon: bool = False
        self._initial_setup: bool = False
        self._high_ground_behavior: bool = True
        self._harass_pylon: BaseCombat = HarassPylon(ai, config, mediator)
        self._high_ground_combat: BaseCombat = HighGroundCombat(ai, config, mediator)
        self._protect_pylon: BaseCombat = ProtectPosition(ai, config, mediator)
        self._unreachable_cells = None
        self._transfused_tags: set[int] = set()

    @property
    def attack_target(self) -> Point2:
        attack_target: Point2 = self.ai.game_info.map_center
        if self.ai.enemy_structures:
            attack_target: Point2 = cy_closest_to(
                self.ai.game_info.map_center, self.ai.enemy_structures
            ).position
        return attack_target

    @property
    def home(self) -> Point2:
        if not self.ai.structures:
            return self.ai.game_info.map_center
        return self.ai.structures[0].position

    @property
    def close_ramp(self) -> bool:
        return (
            cy_distance_to(self.ramps_sorted_to_spawn[0].bottom_center, self.home) < 8.5
        )

    @property
    def defend_position(self) -> Point2:
        if not self.ai.structures:
            return self.ai.game_info.map_center

        if self.close_ramp:
            return self.ramps_sorted_to_spawn[0].top_center
        else:
            return self.ai.structures[0].position

    def _enemy_in_range_of_pylon(self) -> Optional[Units]:
        if everything_near_pylon := self.mediator.get_units_in_range(
            start_points=[self.defend_position],
            distances=8.5,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=False,
        ):
            return everything_near_pylon[0]

    def execute(self):
        map_name: str = self.ai.game_info.map_name.upper()
        # on new micro arena maps
        if (
            "BOT MICRO ARENA" not in map_name and "PLATEAU MICRO" not in map_name
        ) or not self.mediator.get_map_data_object.map_ramps:
            self._new_micro_arena_combat()
        # on the old original micro map with a pylon / ramps
        else:
            if self.ai.structures and self.ai.units and not self._initial_setup:
                self._assign_initial_units()
                self._calculate_high_ground_spots()
                self._initial_setup = True

            if not self._initial_setup:
                return

            self._should_do_high_ground_behavior()

            enemy_threats: Optional[Units] = self._enemy_in_range_of_pylon()
            if enemy_threats and self.ai.get_total_supply(enemy_threats) >= 1.5:
                self._defend_pylon = True
            grid = self.mediator.get_ground_grid

            self._execute_combat(grid)

    def _new_micro_arena_combat(self) -> None:
        self._transfused_tags = set()

        # TODO: stutter forward
        for unit in self.ai.units:
            all_close_enemy: Units = self.mediator.get_units_in_range(
                start_points=[self.ai.units.center],
                distances=100.2,
                query_tree=UnitTreeQueryType.AllEnemy,
            )[0]

            if all_close_enemy:
                maneuver: CombatManeuver = CombatManeuver()
                # work out if all enemy are melee
                all_enemy_melee: bool = all(
                    [
                        u
                        for u in all_close_enemy
                        if not u.is_flying and u.ground_range < 3.0
                    ]
                )
                # melee vs melee fight, kite back weak units
                if (
                    all_enemy_melee
                    and unit.ground_range < 3.0
                    and unit.shield_health_percentage <= self.MELEE_FLEE_AT_PERC
                ):
                    target: Unit = cy_closest_to(unit.position, all_close_enemy)
                    maneuver.add(StutterUnitBack(unit, target))

                # ravager bile
                # TODO: improve
                if (
                    unit.type_id == UnitID.RAVAGER
                    and AbilityId.EFFECT_CORROSIVEBILE in unit.abilities
                ):
                    maneuver.add(
                        UseAbility(
                            AbilityId.EFFECT_CORROSIVEBILE,
                            unit,
                            cy_closest_to(unit.position, all_close_enemy).position,
                        )
                    )

                # queens transfuse
                if (
                    unit.type_id == UnitID.QUEEN
                    and AbilityId.TRANSFUSION_TRANSFUSION in unit.abilities
                ):
                    transfuse_targets: list[Unit] = [
                        u
                        for u in self.ai.units
                        if u.health_percentage < 0.4
                        and cy_distance_to(unit.position, u.position)
                        < 7.0 + unit.radius + u.radius
                        and u.tag != unit.tag
                        and u.tag not in self._transfused_tags
                    ]
                    if transfuse_targets:
                        maneuver.add(
                            UseAbility(
                                AbilityId.TRANSFUSION_TRANSFUSION,
                                unit,
                                transfuse_targets[0],
                            )
                        )

                # catch all that works for most things
                maneuver.add(GenericEngagement(unit, all_close_enemy, False))
                self.ai.register_behavior(maneuver)

    def _execute_combat(self, grid: np.ndarray) -> None:
        defenders: Units = self.mediator.get_units_from_role(role=UnitRole.DEFENDING)
        # if high ground behavior, some units can move out to get vision
        if self._high_ground_behavior:
            high_ranged: list[Unit] = [u for u in defenders if u.type_id in BEST_RANGE]
            remaining: list[Unit] = [
                u for u in defenders if u.type_id not in BEST_RANGE
            ]
            self._high_ground_combat.execute(
                high_ranged,
                grid,
                attack_target=self.attack_target,
                defend_position=self.defend_position,
                move_to=self._close_high_ground_spots[0],
                should_defend=self._defend_pylon,
            )
            self._protect_pylon.execute(
                remaining,
                grid,
                enemy_at_position=self._enemy_in_range_of_pylon(),
                defend_position=self.defend_position,
                should_defend=self._defend_pylon,
            )
        else:
            self._protect_pylon.execute(
                defenders,
                grid,
                enemy_at_position=self._enemy_in_range_of_pylon(),
                defend_position=self.defend_position,
                should_defend=self._defend_pylon,
            )

        self._harass_pylon.execute(
            self.mediator.get_units_from_role(role=UnitRole.HARASSING),
            grid,
            harass_position=self.attack_target,
        )

    def _assign_initial_units(self) -> None:
        assign_attacking: bool = self.ai.race != Race.Protoss and (
            self.ai.race == Race.Zerg or self.ai.enemy_race == Race.Protoss
        )
        if not self._assigned_units and self.ai.units:
            assigned_harass: bool = False
            for i, unit in enumerate(self.ai.units):
                if (
                    assign_attacking
                    and not assigned_harass
                    and unit.type_id not in BEST_RANGE
                ):
                    self.mediator.assign_role(tag=unit.tag, role=UnitRole.HARASSING)
                    assigned_harass = True
                else:
                    self.mediator.assign_role(tag=unit.tag, role=UnitRole.DEFENDING)

                self._assigned_units = True

    def _assigned_units(self) -> None:
        if not self._high_ground_behavior:
            pass

    def _should_do_high_ground_behavior(self):
        if self.close_ramp:
            self._high_ground_behavior = False
            return

        if self._high_ground_behavior and (
            self.ai.corrected_enemy_race == Race.Zerg
            or self._defend_pylon
            or self._enemy_on_high_ground()
        ):
            self._high_ground_behavior = False

    def _enemy_on_high_ground(self) -> bool:
        all_enemy_lower: bool = True
        for enemy in self.ai.enemy_units:
            if self.ai.get_terrain_height(enemy.position) >= self.ai.get_terrain_height(
                self._close_high_ground_spots[0]
            ):
                all_enemy_lower = False
                break
        return not all_enemy_lower

    @property
    def ramps_sorted_to_spawn(self) -> list:
        ramps = self.mediator.get_map_data_object.map_ramps
        return sorted(
            ramps, key=lambda ramp: cy_distance_to(ramp.top_center, self.home)
        )

    def _calculate_high_ground_spots(self) -> None:
        """
        For the two closest ramps, calculate a nearby highground spot
        """
        ramps: list = self.ramps_sorted_to_spawn
        start_spot: Point2 = self.home
        for ramp in ramps[:2]:
            new_start_spot = self.ai.game_info.map_center
            closest = self.ai.game_info.map_center
            dist = 999.9
            for point in ramp.points:
                if self.ai.get_terrain_height(point) > self.ai.get_terrain_height(
                    start_spot
                ):
                    d = cy_distance_to(point, new_start_spot)
                    if d < dist:
                        closest = point
                        dist = d
            self._close_high_ground_spots.append(closest)
