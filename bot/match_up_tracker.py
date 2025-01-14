from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from ares.managers.manager_mediator import ManagerMediator
from loguru import logger
from sc2.ids.unit_typeid import UnitTypeId

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class MatchUpState:
    # inspired from sharpy-micro-p
    mediator: ManagerMediator

    own_unit_tags: set[int] = field(default_factory=set)
    own_unit_types: set[str] = field(default_factory=set)
    enemy_unit_tags: set[int] = field(default_factory=set)
    enemy_unit_types: set[str] = field(default_factory=set)

    start_time: float = 0
    end_time: float = 0

    def init(self):
        for unit in self.mediator.get_own_army:
            self.own_unit_tags.add(unit.tag)
            self.own_unit_types.add(unit.type_id.name)

        for e_unit in self.mediator.get_all_enemy:
            self.enemy_unit_tags.add(e_unit.tag)
            self.enemy_unit_types.add(e_unit.type_id.name)

    def remove_unit_tag(self, tag: int) -> None:
        if tag in self.enemy_unit_tags:
            self.enemy_unit_tags.remove(tag)
        elif tag in self.own_unit_tags:
            self.own_unit_tags.remove(tag)


class MatchUpTracker:

    """
    Track the match up in micro arena, inspired from sharpy-micro-p

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

        self.match_ups: list[MatchUpState] = []
        self.active_match_up: Optional[MatchUpState] = None

    def remove_unit_tag(self, tag: int) -> None:
        if self.active_match_up:
            self.active_match_up.remove_unit_tag(tag)

    async def execute(self) -> None:
        if len(self.ai.units) > 0 and len(self.ai.enemy_units) > 0:
            # Round active
            if not self.active_match_up:
                self.active_match_up = MatchUpState(self.mediator)
                self.active_match_up.start_time = self.ai.time
                self.active_match_up.init()
        else:
            # Round over / not started yet
            if self.active_match_up:
                self.match_ups.append(self.active_match_up)
                self.active_match_up.end_time = self.ai.time
                round_number = len(self.match_ups)
                has_own = len(self.active_match_up.own_unit_tags) > 0
                has_enemy = len(self.active_match_up.enemy_unit_tags) > 0
                formatted_tag = (
                    f"""Tag: R{round_number} """
                    f"""{', '.join(
                        str(unit_type).capitalize()
                        for unit_type in self.active_match_up.own_unit_types
                    )} """
                    f"vs "
                    f"""{', '.join(
                        str(unit_type).capitalize()
                        for unit_type in self.active_match_up.enemy_unit_types
                    )}"""
                )
                await self.ai.chat_send(formatted_tag)
                if (has_enemy and has_own) or (not has_enemy and not has_own):
                    await self.ai.chat_send(f"Tag: Round {round_number} - Tie")
                elif has_own:
                    await self.ai.chat_send(f"Tag: Round {round_number} - Won")
                else:
                    await self.ai.chat_send(f"Tag: Round {round_number} - Lost")

                self.active_match_up = None

                if round_number == 10:
                    own_victories = 0
                    for match_up in self.match_ups:
                        if len(match_up.enemy_unit_tags) == 0:
                            own_victories += 1
                    score_str: str = f"{own_victories} - {round_number - own_victories}"
                    logger.info(score_str)
                    await self.ai.chat_send(f"Tag: {score_str}")
