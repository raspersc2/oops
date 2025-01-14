from typing import TYPE_CHECKING

from ares import ManagerMediator
from ares.behaviors.combat.individual import AttackTarget
from ares.cache import property_cache_once_per_frame
from ares.consts import UnitRole
from cython_extensions import cy_closest_to
from cython_extensions.units_utils import cy_find_units_center_mass
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.units import Units

from bot.combat_squads.main import CombatSquadsController
from bot.match_up_tracker import MatchUpTracker

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

    def __init__(
        self,
        ai: "AresBot",
        config: dict,
        mediator: ManagerMediator,
        match_up_tracker: MatchUpTracker,
    ):
        self.ai: "AresBot" = ai
        self.config: dict = config
        self.mediator: ManagerMediator = mediator
        self.match_up_tracker: MatchUpTracker = match_up_tracker
        self._combat_squad_controller: CombatSquadsController = CombatSquadsController(
            self.ai, self.mediator, UnitRole.ATTACKING
        )

        self._transfused_tags: set[int] = set()
        self._unit_tag_to_bane_tag: dict = dict()
        self._squad_engagement_phase: dict[str, dict] = dict()

    @property_cache_once_per_frame
    def attack_target(self) -> Point2:
        _attack_target: Point2 = self.ai.game_info.map_center
        if self.ai.enemy_units:
            _attack_target = Point2(
                cy_find_units_center_mass(self.ai.enemy_units, 14.0)[0]
            )
        elif self.ai.enemy_structures:
            _attack_target: Point2 = cy_closest_to(
                self.ai.game_info.map_center, self.ai.enemy_structures
            ).position
        return _attack_target

    def execute(self):
        if (
            not self.ai.enemy_units
            or not self.ai.units
            or not self.match_up_tracker.active_match_up
        ):
            return

        self._assign_units_to_banes(self.ai.enemy_units)
        self._combat_squad_controller.execute(
            self.attack_target, self._unit_tag_to_bane_tag
        )

        for unit in self.ai.units:
            if unit.tag in self._unit_tag_to_bane_tag:
                bane_tag: int = self._unit_tag_to_bane_tag[unit.tag]
                if bane := self.ai.unit_tag_dict.get(bane_tag):
                    self.ai.register_behavior(AttackTarget(unit, bane))

    def _assign_units_to_banes(self, all_close_enemy: Units) -> None:
        if not self.ai.units:
            return
        banes: Units = all_close_enemy(UnitID.BANELING)

        for bane in banes:
            assigned_bane_tags: set[int] = set(self._unit_tag_to_bane_tag.values())
            if bane.tag not in assigned_bane_tags:
                for unit in self.ai.units:
                    if (
                        unit.type_id != UnitTypeId.BANELING
                        and unit.is_light
                        and unit.tag not in self._unit_tag_to_bane_tag
                    ):
                        self._unit_tag_to_bane_tag[unit.tag] = bane.tag
                        break
        to_remove: list[int] = []
        for unit_tag in self._unit_tag_to_bane_tag:
            if not self.ai.unit_tag_dict.get(unit_tag):
                to_remove.append(unit_tag)

        for tag in to_remove:
            del self._unit_tag_to_bane_tag[tag]
