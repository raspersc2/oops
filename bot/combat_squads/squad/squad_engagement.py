from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from cython_extensions.combat_utils import cy_attack_ready
from sc2.data import Race

from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.group import AMoveGroup, StutterGroupBack
from ares.behaviors.combat.individual import (
    AMove,
    AttackTarget,
    CombatIndividualBehavior,
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    StutterUnitBack,
    StutterUnitForward,
    UseAbility,
)
from ares.dicts.unit_data import UNIT_DATA
from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from cython_extensions.geometry import cy_distance_to_squared
from cython_extensions.units_utils import cy_closest_to
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat_squads.squad.base_squad import BaseSquad


@dataclass
class SquadEngagement(BaseSquad):
    ai: "AresBot"
    mediator: ManagerMediator
    squad: UnitSquad
    target: Point2
    _transfused_tags: set[int] = field(default_factory=set)
    MELEE_FLEE_AT_PERC: float = 0.3

    def execute(
        self,
        squad: UnitSquad,
        enemy: Union[Units, list[Unit]],
        target: Point2,
        **kwargs,
    ) -> None:
        _unit_tag_to_bane_tag = dict()
        if "_unit_tag_to_bane_tag" in kwargs:
            _unit_tag_to_bane_tag = kwargs["_unit_tag_to_bane_tag"]

        stutter_forward: bool = False
        if "stutter_forward" in kwargs:
            stutter_forward = kwargs["stutter_forward"]

        # no enemy, a-move group and return out of here
        if not enemy:
            self.ai.register_behavior(AMoveGroup(squad.squad_units, squad.tags, target))
            return

        units: list[Unit] = squad.squad_units
        fliers: list[Unit] = [u for u in enemy if UNIT_DATA[u.type_id]["flying"]]
        ground: list[Unit] = [u for u in enemy if not UNIT_DATA[u.type_id]["flying"]]

        # own_fliers: list[Unit] = [u for u in squad.squad_units if UNIT_DATA[u.type_id]["flying"]]
        own_ground: list[Unit] = [
            u for u in squad.squad_units if not UNIT_DATA[u.type_id]["flying"]
        ]

        # all_enemy_low_range: bool = all(
        #     u.ground_range < 3 and u.type_id != UnitID.BANELING for u in ground
        # )
        threshold = 0.85
        count_low_range = sum(1 for u in ground if u.ground_range < 3 and u.type_id != UnitID.BANELING)
        all_enemy_low_range = count_low_range / len(ground) > threshold if ground else False

        all_own_range: bool = all(u for u in own_ground if u.ground_range >= 3.0)

        # micro as a group so we stay together
        do_melee_fight: bool = all_enemy_low_range and (
            all_own_range or self.ai.race != Race.Zerg
        )

        for unit in units:
            avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid
            grid: np.ndarray = self.mediator.get_ground_grid
            if UNIT_DATA[unit.type_id]["flying"]:
                avoid_grid = self.mediator.get_air_avoidance_grid
                grid = self.mediator.get_air_grid

            if do_melee_fight and own_ground:
                total_radius = sum(u.radius for u in own_ground)
                distance_check: float = total_radius / 1.2
                self._fight_vs_melee(unit, ground, grid, squad.squad_position, distance_check)
                continue

            if unit.tag in _unit_tag_to_bane_tag:
                bane_tag: int = _unit_tag_to_bane_tag[unit.tag]
                if bane := self.ai.unit_tag_dict.get(bane_tag):
                    self.ai.register_behavior(AttackTarget(unit, bane))
                    continue

            combat_maneuver: CombatManeuver = CombatManeuver()

            # avoid things like storms, biles etc
            combat_maneuver.add(KeepUnitSafe(unit, avoid_grid))

            # siege, AOE, cyclone lock ons etc etc
            combat_maneuver = self._use_unit_abilities(
                unit, enemy, grid, squad, target, combat_maneuver
            )

            if AbilityId.EFFECT_BLINK_STALKER in unit.abilities and not unit.has_buff(
                BuffId.FUNGALGROWTH
            ):
                safe_spot: Point2 = self.mediator.find_closest_safe_spot(
                    from_pos=unit.position, grid=grid
                )
                combat_maneuver.add(
                    UseAbility(
                        ability=AbilityId.EFFECT_BLINK_STALKER,
                        unit=unit,
                        target=safe_spot,
                    )
                )

            if unit.shield_health_percentage < 0.25:
                combat_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))

            # avoid banes
            if self._should_flee_baneling(unit, enemy):
                combat_maneuver.add(KeepUnitSafe(unit, grid))
            # attack move things if possible
            elif (
                unit.ground_range < 3.0
                and (unit.can_attack or unit.type_id == UnitID.BANELING)
            ) or unit.is_hallucination:
                combat_maneuver = self._melee_attack(
                    combat_maneuver, unit, target, enemy, grid
                )
            # sentry hallucinate units
            elif hallucinate := self._use_hallucinate(unit, target):
                combat_maneuver.add(hallucinate)
            # default attacking logic
            elif unit.can_attack:
                combat_maneuver.add(ShootTargetInRange(unit, ground, extra_range=1.0))
                combat_maneuver.add(ShootTargetInRange(unit, fliers))
                if stutter_forward:
                    _enemy: list[Unit] = (
                        ground if ground else (fliers if fliers else enemy)
                    )
                    combat_maneuver.add(
                        StutterUnitForward(unit, cy_closest_to(unit.position, _enemy))
                    )
                else:
                    combat_maneuver.add(
                        StutterUnitBack(
                            unit, cy_closest_to(unit.position, enemy), grid=grid
                        )
                    )
            else:
                combat_maneuver.add(AMove(unit=unit, target=squad.squad_position))

            self.ai.register_behavior(combat_maneuver)

    @staticmethod
    def _should_flee_baneling(unit, enemy) -> bool:
        return (
            unit.type_id != UnitID.BANELING
            and unit.is_light
            and [
                e
                for e in enemy
                if e.type_id == UnitID.BANELING
                and cy_distance_to_squared(unit.position, e.position) < 15.4
            ]
        )

    def _use_hallucinate(
        self, unit: Unit, target: Point2
    ) -> Optional[CombatIndividualBehavior]:
        if AbilityId.HALLUCINATION_ARCHON in unit.abilities:
            return UseAbility(AbilityId.HALLUCINATION_ARCHON, unit, None)

    def _melee_attack(
        self,
        combat_maneuver: CombatManeuver,
        unit: Unit,
        target: Point2,
        enemy: Units,
        grid: np.ndarray,
    ) -> CombatManeuver:
        # chase down armoured units if zergling
        if unit.type_id == UnitID.ZERGLING:
            if armoured := [
                u for u in enemy if not UNIT_DATA[u.type_id]["flying"] and u.is_armored
            ]:
                target: Unit = cy_closest_to(unit.position, armoured)
                if cy_distance_to_squared(unit.position, target.position) > 16.0:
                    combat_maneuver.add(PathUnitToTarget(unit, grid, target.position))
                else:
                    combat_maneuver.add(AttackTarget(unit=unit, target=target))

        combat_maneuver.add(AMove(unit=unit, target=target))
        return combat_maneuver

    def _fight_vs_melee(
        self,
        u: Unit,
        enemy_ground: list[Unit],
        grid: np.ndarray,
        squad_position: Point2,
        distance_check: float
    ) -> None:
        e_target: Unit = cy_closest_to(squad_position, enemy_ground)
        melee_fight: CombatManeuver = CombatManeuver()

        melee_fight.add(ShootTargetInRange(u, enemy_ground))
        if cy_distance_to_squared(u.position, squad_position) > distance_check:
            melee_fight.add(UseAbility(AbilityId.MOVE_MOVE, u, squad_position))

        if u.ground_range >= 3:
            melee_fight.add(StutterUnitBack(u, e_target, grid=grid))
        else:
            melee_fight.add(AMove(unit=u, target=e_target.position))
        self.ai.register_behavior(melee_fight)
