from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from cython_extensions import cy_is_facing
from sc2.units import Units

from ares import AresBot, UnitRole
from ares.consts import (
    LOSS_DECISIVE_OR_WORSE,
    TIE_OR_BETTER,
    VICTORY_MARGINAL_OR_BETTER,
    EngagementResult,
    UnitTreeQueryType,
)
from ares.dicts.unit_data import UNIT_DATA
from ares.managers.manager_mediator import ManagerMediator
from ares.managers.squad_manager import UnitSquad
from cython_extensions.geometry import cy_distance_to_squared
from cython_extensions.units_utils import cy_center
from loguru import logger
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit

from bot.combat_squads.squad.squad_engagement import SquadEngagement
from bot.combat_squads.squad.squad_movement import SquadMovement
from bot.combat_squads.squad.squad_retreating import SquadRetreating
from bot.combat_squads.squad.squad_setup import SquadSetup

COMBAT_SIM_IGNORE: set[UnitID] = {UnitID.BANELING}
COMMON_UNIT_IGNORE_TYPES: set[UnitID] = {
    UnitID.EGG,
    UnitID.LARVA,
    UnitID.OVERSEER,
    UnitID.OBSERVER,
}


class EngagementPhase(str, Enum):
    # setup formation, move fodder to front etc
    SettingUp = "SettingUp"
    # moving towards target with no enemy around
    Moving = "Moving"
    # after moving and enemy is getting near, and we want the fight
    # should be a very short window to adjust formation
    PreEngaging = "PreEngaging"
    # near enemy, time to fight if we decide to
    Engaging = "Engaging"
    # move away from enemy
    # if we manage to move away from any enemy go
    # back to `SettingUp`
    Retreating = "Retreating"
    # TODO
    # Sieging = "Sieging"


@dataclass
class CombatSquadsController:
    ai: AresBot
    mediator: ManagerMediator
    role: UnitRole
    setup_phase_time: float = 3.0
    pre_engage_setup_time: float = 2.0
    commit_to_engage_for: float = 5.0
    commit_to_disengage_for: float = 3.0
    engage_threshold: set[EngagementResult] = field(default_factory=set)
    disengage_threshold: set[EngagementResult] = field(default_factory=set)
    small_engage_threshold: set[EngagementResult] = field(default_factory=set)
    # for each squad, remember what `SquadEngagementPhase` we are in
    # also store things like what time we changed etc
    _squads_tracker: dict[str, dict] = field(default_factory=dict)
    # for each engagement phase, have a `BaseSquad` object
    # that executes the relevant micro
    _engagement_phase_to_base_squad: dict[EngagementPhase, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.engage_threshold:
            self.engage_threshold = TIE_OR_BETTER
        if not self.disengage_threshold:
            self.disengage_threshold = LOSS_DECISIVE_OR_WORSE
        if not self.small_engage_threshold:
            self.small_engage_threshold = VICTORY_MARGINAL_OR_BETTER

        self._engagement_phase_to_base_squad[EngagementPhase.SettingUp] = SquadSetup
        self._engagement_phase_to_base_squad[EngagementPhase.Moving] = SquadMovement
        self._engagement_phase_to_base_squad[EngagementPhase.PreEngaging] = SquadSetup
        self._engagement_phase_to_base_squad[EngagementPhase.Engaging] = SquadEngagement
        self._engagement_phase_to_base_squad[
            EngagementPhase.Retreating
        ] = SquadRetreating

    def execute(
        self,
        attack_target: Point2,
        _unit_tag_to_bane_tag: dict[int, int],
        squad_radius: float = 9.0,
        close_enemy_radius: float = 14.0,
        far_enemy_radius: float = 18.5,
    ) -> None:
        squads: list[UnitSquad] = self.mediator.get_squads(
            role=self.role, squad_radius=squad_radius
        )

        for squad in squads:
            # we have no info on this squad right now, set things up
            # set things to retreating by default as safest option
            if squad.squad_id not in self._squads_tracker:
                self._add_to_squad_tracker(
                    squad, EngagementPhase.Retreating, self.ai.time, attack_target
                )

            close_enemy: list[Unit] = [
                u
                for u in self.mediator.get_units_in_range(
                    start_points=[squad.squad_position],
                    distances=close_enemy_radius,
                    query_tree=UnitTreeQueryType.AllEnemy,
                )[0]
                if u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]
            max_enemy_range: float = (
                max([u.ground_range for u in close_enemy]) if close_enemy else 0.0
            )
            range_check: float = 4.0 + (max_enemy_range * 1.5)
            super_close_enemy: list[Unit] = [
                u
                for u in self.mediator.get_units_in_range(
                    start_points=[squad.squad_position],
                    distances=range_check,
                    query_tree=UnitTreeQueryType.AllEnemy,
                )[0]
                if u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]

            far_enemy: list[Unit] = [
                u
                for u in self.mediator.get_units_in_range(
                    start_points=[squad.squad_position],
                    distances=far_enemy_radius,
                    query_tree=UnitTreeQueryType.AllEnemy,
                )[0]
                if u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]
            main_fight_should_engage: bool = self._update_squad_engagement(
                squad, squads, close_enemy, far_enemy
            )

            small_fight_should_engage: bool = (
                not main_fight_should_engage
                and self._update_squad_small_engagement(squad, super_close_enemy)
            )

            current_phase: EngagementPhase = self._update_current_squad_phase(
                squad,
                self.ai.time,
                close_enemy,
                far_enemy,
                super_close_enemy,
                main_fight_should_engage,
                small_fight_should_engage,
                attack_target,
            )

            self._track_stutter_forward(squad, far_enemy)

            _move_to: Point2 = (
                attack_target
                if squad.main_squad
                else self.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
            )

            self._execute_squad_control(
                squad,
                current_phase,
                close_enemy,
                super_close_enemy,
                main_fight_should_engage,
                small_fight_should_engage,
                _move_to,
                _unit_tag_to_bane_tag,
            )

    def _execute_squad_control(
        self,
        squad: UnitSquad,
        current_phase: EngagementPhase,
        close_enemy: list[Unit],
        super_close_enemy: list[Unit],
        should_engage: bool,
        small_fight_should_engage: bool,
        attack_target: Point2,
        _unit_tag_to_bane_tag: dict[int, int],
    ) -> None:
        pos_of_main_squad: Point2 = self.mediator.get_position_of_main_squad(
            role=self.role
        )
        self._squads_tracker[squad.squad_id]["combat_object"].execute(
            squad=squad,
            enemy=close_enemy,
            target=attack_target,
            pos_of_main_squad=pos_of_main_squad,
            stutter_forward=self._squads_tracker[squad.squad_id]["stutter_forward"],
            _unit_tag_to_bane_tag=_unit_tag_to_bane_tag,
        )

        if self.ai.config:
            match current_phase:
                case EngagementPhase.SettingUp:
                    self.ai.draw_text_on_world(
                        squad.squad_position, f"{squad.squad_id} Setup Formation"
                    )
                case EngagementPhase.Moving:
                    self.ai.draw_text_on_world(
                        squad.squad_position, f"{squad.squad_id} Moving"
                    )
                case EngagementPhase.PreEngaging:
                    self.ai.draw_text_on_world(
                        squad.squad_position, f"{squad.squad_id} Pre-Engagement"
                    )
                case EngagementPhase.Engaging:
                    self.ai.draw_text_on_world(
                        squad.squad_position, f"{squad.squad_id} Engaging"
                    )
                case EngagementPhase.Retreating:
                    self.ai.draw_text_on_world(
                        squad.squad_position, f"{squad.squad_id} Retreating"
                    )

    def _track_stutter_forward(self, squad: UnitSquad, close_enemy: Units) -> None:
        our_range = [
            u.ground_range
            for u in squad.squad_units
            if not UNIT_DATA[u.type_id]["flying"]
        ]
        our_avg_range = sum(our_range) / len(our_range) if our_range else 0

        enemy_range = [
            u.ground_range for u in close_enemy if not UNIT_DATA[u.type_id]["flying"]
        ]
        enemy_avg_range = sum(enemy_range) / len(enemy_range) if enemy_range else 0

        if our_avg_range < enemy_avg_range:
            self._squads_tracker[squad.squad_id]["stutter_forward"] = True
            self._squads_tracker[squad.squad_id]["time_stutter_set"] = self.ai.time
        else:
            self._squads_tracker[squad.squad_id]["stutter_forward"] = False
            self._squads_tracker[squad.squad_id]["time_stutter_set"] = self.ai.time

    def _reset_engagement(
        self,
        squad: UnitSquad,
        squads: list[UnitSquad],
        switch_to_phase: Optional[EngagementPhase] = None,
    ) -> None:
        """
        Reset engagement back to default settings
        """
        _id: str = squad.squad_id
        self._squads_tracker[_id]["engaging"] = False
        # ensure we turn off the main fight result for all squads
        if squad.main_squad:
            for squad in squads:
                if squad.squad_id not in self._squads_tracker:
                    continue
                self._squads_tracker[squad.squad_id]["main_fight_engage"] = False

        if switch_to_phase:
            self._squads_tracker[_id]["phase"] = switch_to_phase
            self._squads_tracker[_id]["time"] = switch_to_phase

    def _update_current_squad_phase(
        self,
        squad: UnitSquad,
        time: float,
        close_enemy: list[Unit],
        far_enemy: list[Unit],
        super_close_enemy: list[Unit],
        main_fight_engage: bool,
        small_fight_engage: bool,
        target: Point2,
    ) -> EngagementPhase:
        squad_id: str = squad.squad_id
        squad_battle_info: dict = self._squads_tracker[squad_id]
        phase: EngagementPhase = squad_battle_info["phase"]
        switched_time: float = squad_battle_info["time_phase_transition"]
        engage: bool = squad_battle_info["engaging"]
        main_fight: bool = squad_battle_info["main_fight_engage"]
        grid: np.ndarray = self.mediator.get_ground_grid
        # could this decision use RL?
        match phase:
            case EngagementPhase.SettingUp:
                if super_close_enemy or switched_time + self.setup_phase_time < time:
                    self._update_phase_transition(
                        squad, EngagementPhase.Moving, time, target
                    )

            case EngagementPhase.Moving:
                # super close enemy and small fight?
                # or main fight and not main squad? fight right away
                if (
                    small_fight_engage
                    and super_close_enemy
                    or (main_fight_engage and not engage and not squad.main_squad)
                ):
                    self._update_phase_transition(
                        squad, EngagementPhase.Engaging, time, target
                    )
                elif main_fight_engage:
                    if (
                        far_enemy
                        and not super_close_enemy
                        and len(squad.squad_units) > 2
                    ):
                        self._update_phase_transition(
                            squad, EngagementPhase.PreEngaging, time, target
                        )
                    else:
                        self._update_phase_transition(
                            squad, EngagementPhase.Engaging, time, target
                        )
                # really close enemy and still don't want to fight, time to retreat
                elif not (small_fight_engage or main_fight_engage) and any(
                    [
                        not self.mediator.is_position_safe(
                            grid=grid, position=u.position
                        )
                        for u in squad.squad_units
                    ]
                ):
                    self._update_phase_transition(
                        squad, EngagementPhase.Retreating, time, target
                    )

            case EngagementPhase.PreEngaging:
                if (
                    any(
                        [
                            not self.mediator.is_position_safe(
                                grid=grid, position=u.position
                            )
                            for u in squad.squad_units
                        ]
                    )
                    or switched_time + self.pre_engage_setup_time < time
                ):
                    self._update_phase_transition(
                        squad, EngagementPhase.Engaging, time, target
                    )

            case EngagementPhase.Engaging:
                if not main_fight_engage and not small_fight_engage:
                    self._update_phase_transition(
                        squad, EngagementPhase.Retreating, time, target
                    )

            case EngagementPhase.Retreating:
                if super_close_enemy and small_fight_engage:
                    self._update_phase_transition(
                        squad, EngagementPhase.Engaging, time, target
                    )
                # once all units are safe, retreat is complete
                elif all(
                    [
                        self.mediator.is_position_safe(
                            grid=self.mediator.get_ground_grid
                            if not UNIT_DATA[u.type_id]["flying"]
                            else self.mediator.get_air_grid,
                            position=u.position,
                        )
                        for u in squad.squad_units
                    ]
                ):
                    if len(squad.squad_units) > 2:
                        self._update_phase_transition(
                            squad, EngagementPhase.SettingUp, time, target
                        )
                    else:
                        self._update_phase_transition(
                            squad, EngagementPhase.Moving, time, target
                        )

        return self._squads_tracker[squad_id]["phase"]

    def _update_squad_small_engagement(
        self, squad: UnitSquad, enemy: list[Unit]
    ) -> bool:
        # TODO
        return False
        # if not enemy:
        #     self._squads_tracker[squad.squad_id]["small_engagement"] = False
        #     return False
        #
        # previous_decision: bool = self._squads_tracker[squad.squad_id][
        #     "small_engagement"
        # ]
        # squad_units: list[Unit] = squad.squad_units
        # fight_result: EngagementResult = self.mediator.can_win_fight(
        #     own_units=[
        #         u
        #         for u in squad_units
        #         if u.can_attack and u.type_id not in COMBAT_SIM_IGNORE
        #     ],
        #     enemy_units=[
        #         e for e in enemy if e.can_attack and e.type_id not in COMBAT_SIM_IGNORE
        #     ],
        # )
        # engage: bool = fight_result in self.small_engage_threshold
        # if engage != previous_decision:
        #     logger.info(
        #         f"{self.ai.time_formatted} - small engagement changed to {engage}"
        #     )
        #     self._squads_tracker[squad.squad_id]["small_engagement"] = engage
        # return engage

    def _update_squad_engagement(
        self,
        squad: UnitSquad,
        squads: list[UnitSquad],
        close_enemy: list[Unit],
        far_enemy: list[Unit],
    ) -> bool:
        squad_id: str = squad.squad_id

        squad_units: list[Unit] = squad.squad_units
        main_squad: bool = squad.main_squad and len(squad_units) > 7
        squad_battle_info: dict = self._squads_tracker[squad_id]
        main_fight_engage: bool = squad_battle_info["main_fight_engage"]
        engaging: bool = squad_battle_info["engaging"]

        if not far_enemy and engaging:
            self._squads_tracker[squad.squad_id]["engaging"] = False
            self._squads_tracker[squad_id]["time_engagement_switched"] = self.ai.time
            return False

        # if we recently made a new decision, commit to it
        if (
            engaging
            and self.ai.time
            < squad_battle_info["time_engagement_switched"] + self.commit_to_engage_for
        ):
            return True
        # recently decided to disengage here
        elif (
            not main_fight_engage
            and not engaging
            and self.ai.time
            < squad_battle_info["time_engagement_switched"]
            + self.commit_to_disengage_for
        ):
            return False

        # the main squad is currently controlling the decision
        if not main_squad and main_fight_engage:
            return True

        enemy: list[Unit] = [
            e for e in far_enemy if e.can_attack and e.type_id not in COMBAT_SIM_IGNORE
        ]
        fight_result: EngagementResult
        if all([e for e in enemy if not e.can_attack]):
            fight_result = EngagementResult.VICTORY_EMPHATIC
        else:
            fight_result = self.mediator.can_win_fight(
                own_units=[
                    u
                    for u in squad_units
                    if u.can_attack and u.type_id not in COMBAT_SIM_IGNORE
                ],
                enemy_units=enemy,
            )

        # currently engaging and we should disengage
        if engaging and fight_result in self.disengage_threshold:
            self._squads_tracker[squad_id]["engaging"] = False
            self._squads_tracker[squad_id]["time_engagement_switched"] = self.ai.time

            if main_squad:
                logger.info(f"{self.ai.time_formatted} Main fight disengaging")
                self._squads_tracker[squad_id]["main_fight_engage"] = False

        # not engaging and we should engage
        elif not engaging and fight_result in self.engage_threshold and far_enemy:
            self._squads_tracker[squad_id]["engaging"] = True
            self._squads_tracker[squad_id]["time_engagement_switched"] = self.ai.time

            if main_squad:
                logger.info(f"{self.ai.time_formatted} Main fight engaging")
                self._squads_tracker[squad_id]["main_fight_engage"] = True

        # if main squad, update other squads nearby about main fight happening
        if main_squad and far_enemy:
            _engaging: bool = self._squads_tracker[squad_id]["engaging"]
            enemy_position: tuple[float, float] = cy_center(far_enemy)
            for squad in squads:
                if (
                    squad.squad_id in self._squads_tracker
                    and not squad.main_squad
                    and cy_distance_to_squared(squad.squad_position, enemy_position)
                    < 400.0
                ):
                    self._squads_tracker[squad.squad_id][
                        "main_fight_engage"
                    ] = _engaging

        return (
            self._squads_tracker[squad_id]["main_fight_engage"]
            or self._squads_tracker[squad_id]["engaging"]
        )

    def _add_to_squad_tracker(
        self,
        squad: UnitSquad,
        phase: EngagementPhase,
        time_phase_transition: float,
        target: Point2,
        engagement_result: EngagementResult = EngagementResult.LOSS_EMPHATIC,
        stutter_forward: bool = False,
        time_stutter_set: float = 0.0,
        engaging: bool = False,
        time_engagement_switched: float = 0.0,
        small_engagement: bool = False,
        main_fight_engage: bool = False,
    ) -> None:
        self._squads_tracker[squad.squad_id] = {
            "combat_object": self._engagement_phase_to_base_squad[phase](
                self.ai, self.mediator, squad, target
            ),
            "phase": phase,
            "time_phase_transition": time_phase_transition,
            "engagement_result": engagement_result,
            "stutter_forward": stutter_forward,
            "time_stutter_set": time_stutter_set,
            "engaging": engaging,
            "time_engagement_switched": time_engagement_switched,
            "small_engagement": small_engagement,
            "main_fight_engage": main_fight_engage,
        }

    def _update_phase_transition(
        self, squad: UnitSquad, phase: EngagementPhase, time: float, target: Point2
    ) -> None:
        squad_id: str = squad.squad_id
        # ensure the previous object is removed
        del self._squads_tracker[squad_id]["combat_object"]

        self._squads_tracker[squad_id][
            "combat_object"
        ] = self._engagement_phase_to_base_squad[phase](
            self.ai, self.mediator, squad, target
        )
        self._squads_tracker[squad_id]["phase"] = phase
        self._squads_tracker[squad_id]["time_phase_transition"] = time
