from typing import Optional

from ares import AresBot
from ares.consts import ALL_STRUCTURES
from ares.cython_extensions.units_utils import cy_closest_to
from ares.dicts.unit_data import UNIT_DATA
from sc2.data import Race
from sc2.position import Point2
from sc2.units import Units

from bot.combat_manager import CombatManager


class MyBot(AresBot):
    combat_manager: CombatManager

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
        self._sent_oops_chat: bool = False
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

    @property
    def corrected_enemy_race(self) -> Race:
        if self.enemy_race != Race.Random:
            return self.enemy_race

        # random enemy race, update based on first enemy unit
        if not self._detected_race and self.enemy_units:
            self._detected_enemy_race = self.enemy_units[0].race

        return self._detected_enemy_race

    async def on_start(self) -> None:
        await super(MyBot, self).on_start()
        self.combat_manager = CombatManager(self, self.config, self.mediator)

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)

        self.combat_manager.execute()

        if not self._sent_oops_chat and not self.units and self.time > 10.0:
            await self.chat_send("oops")
            self._sent_oops_chat = True

        if not self._sent_race_tag and self.time > 5.0:
            await self.chat_send(f"Tag: {self.race}", True)
            self._sent_race_tag = True

    def get_total_supply(self, units: Units) -> int:
        return sum(
            [
                UNIT_DATA[unit.type_id]["supply"]
                for unit in units
                if unit.type_id not in ALL_STRUCTURES
            ]
        )
