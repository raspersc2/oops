from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.group import AMoveGroup
from ares.behaviors.combat.individual import (
    AMove,
    AttackTarget,
    CombatIndividualBehavior,
    KeepUnitSafe,
    ShootTargetInRange,
    StutterUnitBack,
    UseAbility,
    UseTransfuse,
)
from ares.behaviors.combat.individual.ghost_snipe import GhostSnipe
from ares.behaviors.combat.individual.medivac_heal import MedivacHeal
from ares.behaviors.combat.individual.raven_auto_turret import RavenAutoTurret
from ares.behaviors.combat.individual.siege_tank_decision import SiegeTankDecision
from ares.behaviors.combat.individual.use_aoe_ability import UseAOEAbility
from ares.dicts.aoe_ability_to_range import AOE_ABILITY_SPELLS_INFO
from ares.dicts.unit_data import UNIT_DATA
from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from cython_extensions.geometry import cy_distance_to_squared
from cython_extensions.units_utils import cy_closest_to
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat_squads.squad.base_squad import BaseSquad
from bot.combat_squads.squad.feed_back import FeedBack


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

        # no enemy, a-move group and return out of here
        if not enemy:
            self.ai.register_behavior(AMoveGroup(squad.squad_units, squad.tags, target))
            return

        units: list[Unit] = squad.squad_units

        for unit in units:
            avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid
            grid: np.ndarray = self.mediator.get_ground_grid
            if UNIT_DATA[unit.type_id]["flying"]:
                avoid_grid = self.mediator.get_air_avoidance_grid
                grid = self.mediator.get_air_grid

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

            if (
                unit.shield_percentage < 0.2
                and AbilityId.EFFECT_BLINK_STALKER in unit.abilities
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

            # avoid banes
            if self._should_flee_baneling(unit, enemy):
                combat_maneuver.add(KeepUnitSafe(unit, grid))
            # attack move things if possible
            elif (
                unit.ground_range < 3.0
                and (unit.can_attack or unit.type_id == UnitID.BANELING)
            ) or unit.is_hallucination:
                combat_maneuver.add(AMove(unit=unit, target=target))
            # sentry hallucinate units
            elif hallucinate := self._use_hallucinate(unit, target):
                combat_maneuver.add(hallucinate)
            # default attacking logic
            elif unit.can_attack:
                combat_maneuver.add(ShootTargetInRange(unit, enemy))
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

    def _use_aoe_ability(
        self, unit: Unit, enemy: list[Unit]
    ) -> Optional[CombatIndividualBehavior]:
        for ability in AOE_ABILITY_SPELLS_INFO:
            if ability not in unit.abilities:
                continue

            ability_range_squared: float = 9.0 + (
                AOE_ABILITY_SPELLS_INFO[ability]["range"] ** 2
            )
            _targets: list[Unit] = Units(
                [
                    u
                    for u in enemy
                    if cy_distance_to_squared(u.position, unit.position)
                    <= ability_range_squared
                ],
                self.ai,
            )
            if _targets:
                min_targets: int = 4
                if ability in {
                    AbilityId.EFFECT_CORROSIVEBILE,
                    AbilityId.KD8CHARGE_KD8CHARGE,
                }:
                    min_targets = 1
                avoid_own_ground: bool = ability in {
                    AbilityId.KD8CHARGE_KD8CHARGE,
                    AbilityId.PSISTORM_PSISTORM,
                }
                avoid_own_flying: bool = ability in {AbilityId.PSISTORM_PSISTORM}
                return UseAOEAbility(
                    unit,
                    ability,
                    _targets,
                    min_targets=min_targets,
                    avoid_own_ground=avoid_own_ground,
                    avoid_own_flying=avoid_own_flying,
                )

    def _use_hallucinate(
        self, unit: Unit, target: Point2
    ) -> Optional[CombatIndividualBehavior]:
        if AbilityId.HALLUCINATION_ARCHON in unit.abilities:
            return UseAbility(AbilityId.HALLUCINATION_ARCHON, unit, None)

    def _use_unit_abilities(
        self, unit, enemy, grid, squad, target, combat_maneuver
    ) -> CombatManeuver:
        if self.mediator.is_position_safe(grid=grid, position=unit.position):
            combat_maneuver.add(GhostSnipe(unit, enemy))
        combat_maneuver.add(FeedBack(unit, enemy, 4.5))
        if aoe_ability := self._use_aoe_ability(unit, enemy):
            combat_maneuver.add(aoe_ability)

        # combat_maneuver.add(SiegeTankDecision(unit, enemy, target))
        combat_maneuver.add(RavenAutoTurret(unit, enemy))
        combat_maneuver.add(MedivacHeal(unit, squad.squad_units, grid, keep_safe=False))

        combat_maneuver.add(UseTransfuse(unit, squad.squad_units, extra_range=1.5))

        return combat_maneuver
