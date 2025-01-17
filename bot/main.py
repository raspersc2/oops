from typing import Optional

from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

from ares import AresBot, UnitRole
from ares.consts import ALL_STRUCTURES
from ares.dicts.unit_data import UNIT_DATA
from cython_extensions.units_utils import cy_closest_to
from sc2.data import Race
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat_manager import CombatManager
from bot.match_up_tracker import MatchUpTracker


class MyBot(AresBot):
    combat_manager: CombatManager
    match_up_tracker: MatchUpTracker

    def __init__(self, game_step_override: Optional[int] = None):
        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)

        self._detected_race: bool = False
        self._detected_enemy_race: Race = Race.Random
        self._sent_race_tag: bool = False
        self._unreachable_cells = None

    @property
    def attack_target(self) -> Point2:
        attack_target: Point2 = self.game_info.map_center
        if self.enemy_structures:
            attack_target: Point2 = cy_closest_to(
                self.game_info.map_center, self.enemy_structures
            ).position
        return attack_target

    async def on_start(self) -> None:
        await super(MyBot, self).on_start()
        self.match_up_tracker = MatchUpTracker(self, self.config, self.mediator)
        self.combat_manager = CombatManager(
            self, self.config, self.mediator, self.match_up_tracker
        )

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)

        self.combat_manager.execute()

        await self.match_up_tracker.execute()

        if not self._sent_race_tag and self.time > 5.0:
            await self.chat_send(f"Tag: My Race: {self.race.name}", True)
            await self.chat_send(f"Tag: Enemy Race: {self.enemy_race.name}", True)
            self._sent_race_tag = True

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        await super(MyBot, self).on_unit_destroyed(unit_tag)
        self.match_up_tracker.remove_unit_tag(unit_tag)

    async def on_unit_created(self, unit: Unit) -> None:
        # on micro ladder, assign all to attacking by default
        self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
        if unit.type_id == UnitTypeId.CYCLONE:
            await self.client.toggle_autocast([unit], AbilityId.LOCKON_LOCKON)

    def get_total_supply(self, units: Units) -> int:
        return sum(
            [
                UNIT_DATA[unit.type_id]["supply"]
                for unit in units
                if unit.type_id not in ALL_STRUCTURES
            ]
        )
