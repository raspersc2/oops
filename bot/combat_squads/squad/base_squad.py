from typing import TYPE_CHECKING

from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from sc2.position import Point2
from sc2.unit import Unit

from bot.combat_squads.consts import FODDER_VALUES

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
