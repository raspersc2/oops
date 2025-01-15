from typing import TYPE_CHECKING, Optional

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    CombatIndividualBehavior,
    GhostSnipe,
    MedivacHeal,
    RavenAutoTurret,
    UseAOEAbility,
    UseTransfuse,
)
from ares.dicts.aoe_ability_to_range import AOE_ABILITY_SPELLS_INFO
from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_distance_to_squared
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat_squads.consts import FODDER_VALUES
from bot.combat_squads.squad.feed_back import FeedBack

if TYPE_CHECKING:
    from ares import AresBot


class BaseSquad:
    """Base class for all squad combat classes."""

    def __init__(
        self, ai: "AresBot", mediator: ManagerMediator, squad: UnitSquad, target: Point2
    ):
        self.ai = ai
        self.mediator = mediator
        self.squad = squad
        self.target = target

    def set_squad(self, squad: UnitSquad) -> None:
        self.squad = squad

    def set_target(self, target: Point2) -> None:
        self.target = target

    @staticmethod
    def get_fodder_tags(units: list[Unit]) -> set[int]:
        unit_type_fodder_values: set[int] = {
            FODDER_VALUES[u.type_id] for u in units if u.type_id in FODDER_VALUES
        }

        fodder_tags: set[int] = set()

        if len(unit_type_fodder_values) > 1:
            for unit in units:
                if unit.type_id in FODDER_VALUES and FODDER_VALUES[unit.type_id] == min(
                    unit_type_fodder_values
                ):
                    fodder_tags.add(unit.tag)

        return fodder_tags

    def _use_aoe_ability(
        self, unit: Unit, enemy: list[Unit]
    ) -> Optional[CombatIndividualBehavior]:
        for ability in AOE_ABILITY_SPELLS_INFO:
            if ability not in unit.abilities:
                continue

            if ability == AbilityId.EMP_EMP and self.ai.enemy_race != Race.Protoss:
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
