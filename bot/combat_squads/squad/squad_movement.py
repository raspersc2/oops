from dataclasses import dataclass
from typing import Union

from ares import AresBot
from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_adjust_moving_formation
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat_squads.squad.base_squad import BaseSquad


@dataclass
class SquadMovement(BaseSquad):
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
        units: list[Unit] = squad.squad_units
        fodder_tags: list[int] = list(self.get_fodder_tags(units))
        need_to_move: dict[int, tuple[float, float]] = dict()
        if len(fodder_tags) > 0:
            need_to_move = cy_adjust_moving_formation(
                squad.squad_units, target, fodder_tags, 1.9, 0.25
            )

        for unit in units:
            if unit.tag in need_to_move:
                unit.move(Point2(need_to_move[unit.tag]))
            else:
                unit.move(target)
