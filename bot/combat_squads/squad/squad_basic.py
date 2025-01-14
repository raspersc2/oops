from dataclasses import dataclass
from typing import Union

from ares import AresBot
from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat_squads.squad.base_squad import BaseSquad


@dataclass
class SquadBasic(BaseSquad):
    """
    Basic squad class used for testing
    """

    ai: "AresBot"
    mediator: ManagerMediator
    squad: UnitSquad
    target: Point2

    def execute(
        self,
        squad: UnitSquad,
        enemy: Union[Units, list[Unit]],
        target: Point2,
        **kwargs
    ) -> None:
        for unit in squad.squad_units:
            unit.attack(target)
