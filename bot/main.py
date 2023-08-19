from collections import deque
from typing import Optional

import numpy as np
from ares import AresBot
from ares.consts import ALL_STRUCTURES
from ares.dicts.unit_data import UNIT_DATA
from sc2.data import Race
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.vs_generic import VsGeneric


class MyBot(AresBot):
    combat_controller: BaseCombat

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
        self._unreachable_cells = None

    @property
    def corrected_enemy_race(self) -> Race:
        if self.enemy_race != Race.Random:
            return self.enemy_race

        # random enemy race, update based on first enemy unit
        if not self._detected_race and self.enemy_units:
            self._detected_enemy_race = self.enemy_units[0].race
            self.choose_combat_controller(self._detected_enemy_race)

        return self._detected_enemy_race

    async def on_start(self) -> None:
        await super(MyBot, self).on_start()
        self.choose_combat_controller(self.race)

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)

        # temporary fix to pathing grid so we don't try to reach unpathable areas
        grid = self._fix_grid(self.mediator.get_ground_grid.copy())

        self.combat_controller.execute(self.units, grid)

        if not self._sent_oops_chat and not self.units and self.time > 10.0:
            await self.chat_send("oops")
            self._sent_oops_chat = True

        for unit in self.units:
            self.draw_text_on_world(unit.position, f"{unit.tag}")

    def _fix_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Temporary fix to our pathing grid, since there are
        a bunch of high ground pathable cells we can't reach.
        Which messes with safe finding checks.
        """
        # after the first run, results are cached
        if self._unreachable_cells is not None:
            grid[self._unreachable_cells] = np.inf
            return grid

        start = self.game_info.map_center.rounded
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        queue = deque([start])
        visited[start[0], start[1]] = True

        while queue:
            current_row, current_col = queue.popleft()

            # Define the possible neighboring cells (up, down, left, right)
            neighbors = [
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ]

            for neighbor_row, neighbor_col in neighbors:
                if (
                    0 <= neighbor_row < rows
                    and 0 <= neighbor_col < cols
                    and grid[neighbor_row, neighbor_col] == 1
                    and not visited[neighbor_row, neighbor_col]
                ):
                    visited[neighbor_row, neighbor_col] = True
                    queue.append((neighbor_row, neighbor_col))

        unreachable_cells = np.logical_and(grid == 1, np.logical_not(visited))
        self._unreachable_cells = unreachable_cells
        grid[unreachable_cells] = np.inf
        return grid

    def choose_combat_controller(self, race: Race) -> None:
        self.combat_controller = VsGeneric(self, self.config, self.mediator)

    def get_total_supply(self, units: Units) -> int:
        return sum(
            [
                UNIT_DATA[unit.type_id]["supply"]
                for unit in units
                if unit.type_id not in ALL_STRUCTURES
            ]
        )
