import asyncio
import io
import unittest
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import db as database


# api.py는 import 시 create_all을 실행하므로 테스트용 메모리 DB를 먼저 주입한다.
_IMPORT_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
database.engine = _IMPORT_ENGINE
database.SessionLocal.configure(bind=_IMPORT_ENGINE)

import api  # noqa: E402
from fatigue_store import seed_initial_fatigue_models  # noqa: E402
from fatigue_dataset import parse_fatigue_dataset  # noqa: E402
from fatigue_model import (  # noqa: E402
    WORKER_RECOVERY_HALF_LIFE_MINUTES,
    fatigue_decay,
    recovered_daily_load,
    recovery_minutes_since_previous,
    with_recovery_features,
)
from models import Base, FatigueModel, Organization, Schedule, StaffingDecision, User, WorkSession  # noqa: E402
from worker_personalization import provision_personalized_workers  # noqa: E402


class WorkTrackingApiTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.organization = Organization(name="관재팀", entry_code="ABC234")
        self.db.add(self.organization)
        self.db.flush()
        self.user = User(
            username="tracking-user",
            email="tracking@example.com",
            full_name="김작업",
            password_hash="test-only",
            organization_id=self.organization.id,
            role="worker",
        )
        self.admin = User(
            username="admin-user",
            email="admin@example.com",
            full_name="관리자",
            password_hash="test-only",
            organization_id=self.organization.id,
            role="admin",
        )
        self.db.add_all([self.user, self.admin])
        self.db.commit()
        self.db.refresh(self.user)
        self.db.refresh(self.admin)
        self.original_today_kst = api.today_kst
        self.original_now_kst = api.now_kst
        api.today_kst = lambda: date(2026, 8, 19)
        api.now_kst = lambda: datetime(2026, 8, 19, 10, 0, tzinfo=api.KST)

    def tearDown(self):
        api.today_kst = self.original_today_kst
        api.now_kst = self.original_now_kst
        self.db.close()
        self.engine.dispose()

    def test_work_session_is_idempotent_and_builds_daily_load(self):
        schedules = [
            Schedule(
                organization_id=self.organization.id,
                신청번호=f"A-{schedule_id}",
                출동일시=datetime(2026, 8, 19, 9, 0),
                품명="책상",
                설치장소="공학관 101호",
                필요인원수=2,
                투입인원수=2,
            )
            for schedule_id in (1, 2)
        ]
        self.db.add_all(schedules)
        self.db.flush()
        self.db.add(StaffingDecision(
            organization_id=self.organization.id,
            confirmed_by_user_id=self.admin.id,
            dispatch_time=datetime(2026, 8, 19, 9, 0),
            schedule_ids=[schedule.id for schedule in schedules],
            selected_workers=[self.user.full_name],
            confirmed_at=datetime(2026, 8, 19, 8, 30),
        ))
        self.db.commit()
        body = api.WorkSessionCreate(
            client_session_id="dispatch-1",
            worker_name="김작업",
            schedule_ids=[schedules[1].id, schedules[0].id, schedules[1].id],
            application_numbers=["A-1", "A-1"],
            started_at=datetime(2026, 8, 19, 9, 0, tzinfo=api.KST),
            completed_at=datetime(2026, 8, 19, 9, 40, tzinfo=api.KST),
            total_seconds=2400,
            work_seconds=1800,
            driving_seconds=600,
            unknown_seconds=0,
            gps_sample_count=30,
            gps_rejected_count=1,
            tracking_quality="estimated",
            borg_cr10=4,
        )

        first = api.create_work_session(body, self.user, self.db)
        second = api.create_work_session(body, self.user, self.db)

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["user_id"], self.user.id)
        self.assertEqual(self.db.query(WorkSession).count(), 1)
        self.assertEqual(first["schedule_ids"], sorted(schedule.id for schedule in schedules))
        status = api.workers_status_today(self.admin, self.db)[0]
        self.assertEqual(status["latest_borg_source"], "user")
        self.assertEqual(status["total_work_seconds"], 1800)
        self.assertEqual(status["daily_load"], 3.564)
        self.assertEqual(status["cumulative_fatigue"], 3.564)
        self.assertTrue(status["recovery_applied"])
        self.assertEqual(status["recovery_minutes_since_latest"], 20.0)
        self.assertEqual(first["feature_snapshot"]["cumulative_fatigue"], 0)
        self.assertEqual(first["feature_snapshot"]["recovery_minutes"], 0)

    def test_staffing_preview_does_not_mutate_schedule_and_confirmation_is_explicit(self):
        dispatch_time = datetime(2026, 8, 19, 14, 0)
        schedule = Schedule(
            organization_id=self.organization.id,
            신청번호="A-2",
            출동일시=dispatch_time,
            품명="책상",
            설치장소="공학관 101호",
            신청부서="시설팀",
            수량=1,
            필요인원수=2,
            투입인원수=2,
            가용명단="김작업, 이작업, 박작업",
            출동확정=False,
        )
        self.db.add(schedule)
        self.db.commit()
        self.db.refresh(schedule)

        proposals = api.staffing_recommendations_today(self.admin, self.db)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["team_size"], 2)
        self.assertFalse(self.db.get(Schedule, schedule.id).출동확정)
        self.assertEqual(self.db.query(StaffingDecision).count(), 0)

        selected = ["이작업", "박작업"]
        result = api.confirm_staffing_recommendation(
            api.StaffingDecisionCreate(
                dispatch_time=dispatch_time,
                schedule_ids=[schedule.id],
                selected_workers=selected,
            ),
            self.admin,
            self.db,
        )

        self.assertEqual(result["selected_workers"], selected)
        self.assertEqual(self.db.query(StaffingDecision).count(), 1)
        self.assertFalse(self.db.get(Schedule, schedule.id).출동확정)

    def test_admin_navigation_progress_is_shared_only_with_assigned_workers(self):
        dispatch_time = datetime(2026, 8, 19, 14, 0)
        schedule = Schedule(
            organization_id=self.organization.id,
            신청번호="NAV-1",
            출동일시=dispatch_time,
            품명="책상",
            설치장소="공학관 101호",
        )
        unassigned = User(
            username="unassigned-worker",
            email="unassigned@example.com",
            full_name="미배정",
            password_hash="test-only",
            organization_id=self.organization.id,
            role="worker",
        )
        self.db.add_all([schedule, unassigned])
        self.db.flush()
        decision = StaffingDecision(
            organization_id=self.organization.id,
            confirmed_by_user_id=self.admin.id,
            dispatch_time=dispatch_time,
            schedule_ids=[schedule.id],
            selected_workers=[self.user.full_name],
            recommendation_snapshot={"basis": "test"},
            confirmed_at=datetime(2026, 8, 19, 13, 30),
        )
        self.db.add(decision)
        self.db.commit()

        initial = api.get_navigation_progress(dispatch_time, self.user, self.db)
        self.assertEqual(initial["phase"], "overview")
        self.assertEqual(initial["revision"], 0)

        updated = api.update_navigation_progress(
            api.NavigationProgressUpdate(
                dispatch_time=dispatch_time,
                phase="nav",
                active_schedule_id=schedule.id,
                step_index=3,
            ),
            self.admin,
            self.db,
        )
        worker_view = api.get_navigation_progress(dispatch_time, self.user, self.db)
        self.assertEqual(worker_view, updated)
        self.assertEqual(worker_view["step_index"], 3)
        self.assertEqual(decision.recommendation_snapshot["basis"], "test")

        with self.assertRaises(api.HTTPException) as denied:
            api.get_navigation_progress(dispatch_time, unassigned, self.db)
        self.assertEqual(denied.exception.status_code, 403)

    def test_personal_model_becomes_ready_after_stable_validation(self):
        for index in range(8):
            minutes = 20 + index * 5
            self.db.add(WorkSession(
                organization_id=self.organization.id,
                user_id=self.user.id,
                client_session_id=f"history-{index}",
                worker_name=self.user.full_name,
                schedule_ids=[],
                application_numbers=[],
                started_at=datetime(2026, 8, 18, 9, 0),
                completed_at=datetime(2026, 8, 18, 9, minutes),
                total_seconds=minutes * 60,
                work_seconds=minutes * 60,
                driving_seconds=0,
                unknown_seconds=0,
                gps_sample_count=10,
                gps_rejected_count=0,
                tracking_quality="estimated",
                borg_cr10=4,
                borg_source="user",
                predicted_borg_cr10=None if index < 3 else 4.2,
                prediction_confidence="medium",
                prediction_model_version="personal-ridge-v1",
                feature_snapshot={
                    "work_minutes": minutes,
                    "total_minutes": minutes,
                    "driving_minutes": 0,
                    "unknown_minutes": 0,
                    "item_count": 0,
                    "labor_load": 0,
                    "team_size": 0,
                },
            ))
        self.db.commit()

        result = api.predict_fatigue_after_work(
            api.FatiguePredictionRequest(
                schedule_ids=[],
                total_seconds=3600,
                work_seconds=3600,
                driving_seconds=0,
                unknown_seconds=0,
            ),
            self.user,
            self.db,
        )
        self.assertTrue(result["model_ready"])
        self.assertFalse(result["survey_required"])
        self.assertAlmostEqual(result["predicted_borg_cr10"], 4.0, places=1)

    def test_worker_status_and_schedule_are_isolated_by_organization(self):
        other_org = Organization(name="다른 조직", entry_code="XYZ789")
        self.db.add(other_org)
        self.db.flush()
        other_user = User(
            username="other-user",
            email="other@example.com",
            full_name="다른작업자",
            password_hash="test-only",
            organization_id=other_org.id,
            role="worker",
        )
        own_schedule = Schedule(
            organization_id=self.organization.id,
            신청번호="OWN-1",
            출동일시=datetime(2026, 8, 19, 14, 0),
            품명="책상",
            설치장소="공학관 101호",
        )
        other_schedule = Schedule(
            organization_id=other_org.id,
            신청번호="OTHER-1",
            출동일시=datetime(2026, 8, 19, 14, 0),
            품명="의자",
            설치장소="공학관 102호",
        )
        self.db.add_all([other_user, own_schedule, other_schedule])
        self.db.flush()
        self.db.add(StaffingDecision(
            organization_id=self.organization.id,
            confirmed_by_user_id=self.admin.id,
            dispatch_time=own_schedule.출동일시,
            schedule_ids=[own_schedule.id],
            selected_workers=[self.user.full_name],
            confirmed_at=datetime(2026, 8, 19, 12, 0),
        ))
        self.db.add(WorkSession(
            organization_id=other_org.id,
            user_id=other_user.id,
            client_session_id="other-session",
            worker_name=other_user.full_name,
            schedule_ids=[other_schedule.id],
            application_numbers=["OTHER-1"],
            started_at=datetime(2026, 8, 19, 13, 0),
            completed_at=datetime(2026, 8, 19, 13, 30),
            total_seconds=1800,
            work_seconds=1800,
            driving_seconds=0,
            unknown_seconds=0,
            tracking_quality="estimated",
            borg_cr10=8,
            borg_source="user",
        ))
        self.db.commit()

        schedules = api.schedules_today(self.user, self.db)
        statuses = api.workers_status_today(self.admin, self.db)
        self.assertEqual([row["신청번호"] for row in schedules], ["OWN-1"])
        self.assertEqual([row["worker_name"] for row in statuses], ["김작업"])

    def test_initial_test_status_is_visible_and_admin_is_not_assignable(self):
        seed_initial_fatigue_models(self.db, self.organization.id)
        dispatch_time = datetime(2026, 8, 19, 14, 0)
        self.db.add(Schedule(
            organization_id=self.organization.id,
            신청번호="SEED-1",
            출동일시=dispatch_time,
            품명="책상",
            설치장소="공학관 101호",
            필요인원수=2,
            투입인원수=2,
            가용명단="김경언, 관리자, 강경래",
        ))
        self.db.commit()

        statuses = api.workers_status_today(self.admin, self.db)
        status_by_name = {row["worker_name"]: row for row in statuses}
        self.assertNotIn("관리자", status_by_name)
        self.assertEqual(status_by_name["김경언"]["state_source"], "test_seed")
        self.assertGreater(status_by_name["김경언"]["state_borg_cr10"], 0)

        proposal = api.staffing_recommendations_today(self.admin, self.db)[0]
        self.assertNotIn("관리자", proposal["available_workers"])
        self.assertEqual(proposal["team_size"], 2)

    def test_worker_sees_only_latest_manager_assignment(self):
        other_worker = User(
            username="second-worker",
            email="second-worker@example.com",
            full_name="이작업",
            password_hash="test-only",
            organization_id=self.organization.id,
            role="worker",
        )
        schedule = Schedule(
            organization_id=self.organization.id,
            신청번호="ASSIGN-1",
            출동일시=datetime(2026, 8, 19, 14, 0),
            품명="책상",
            설치장소="공학관 101호",
            필요인원수=1,
            투입인원수=1,
            가용명단="김작업, 이작업",
        )
        self.db.add_all([other_worker, schedule])
        self.db.flush()
        self.assertEqual(len(api.schedules_today(self.admin, self.db)), 1)
        self.assertEqual(api.schedules_today(self.user, self.db), [])

        self.db.add(StaffingDecision(
            organization_id=self.organization.id,
            confirmed_by_user_id=self.admin.id,
            dispatch_time=schedule.출동일시,
            schedule_ids=[schedule.id],
            selected_workers=[self.user.full_name],
            confirmed_at=datetime(2026, 8, 19, 12, 0),
        ))
        self.db.commit()
        assigned = api.schedules_today(self.user, self.db)
        self.assertEqual([row["id"] for row in assigned], [schedule.id])
        self.assertEqual(assigned[0]["배정인원"], [self.user.full_name])

        self.db.add(StaffingDecision(
            organization_id=self.organization.id,
            confirmed_by_user_id=self.admin.id,
            dispatch_time=schedule.출동일시,
            schedule_ids=[schedule.id],
            selected_workers=[other_worker.full_name],
            confirmed_at=datetime(2026, 8, 19, 12, 5),
        ))
        self.db.commit()
        self.assertEqual(api.schedules_today(self.user, self.db), [])
        self.assertEqual([row["id"] for row in api.schedules_today(other_worker, self.db)], [schedule.id])

    def test_personal_model_parameters_are_persisted_after_three_actual_responses(self):
        for index in range(3):
            body = api.WorkSessionCreate(
                client_session_id=f"model-save-{index}",
                worker_name=self.user.full_name,
                schedule_ids=[],
                application_numbers=[],
                started_at=datetime(2026, 8, 19, 9 + index, 0),
                completed_at=datetime(2026, 8, 19, 9 + index, 20),
                total_seconds=1200,
                work_seconds=900,
                driving_seconds=300,
                unknown_seconds=0,
                gps_sample_count=10,
                gps_rejected_count=0,
                tracking_quality="estimated",
                borg_cr10=3 + index,
            )
            api.create_work_session(body, self.user, self.db)

        model = self.db.query(FatigueModel).filter(
            FatigueModel.user_id == self.user.id,
            FatigueModel.scope == "personal",
        ).one()
        self.assertEqual(model.source, "operational")
        self.assertEqual(model.actual_response_count, 3)
        self.assertEqual(len(model.parameters["coefficients"]), 10)
        self.assertIn("cumulative_fatigue", model.feature_schema)
        self.assertIn("recovery_minutes", model.feature_schema)

    def test_uploaded_dataset_trains_global_and_personal_models_immediately(self):
        csv_text = """작업자,작업시간,총시간,차량이동시간,미분류시간,물품개수,노동량,투입인원수,예상피로도,메모
김작업,20,30,8,2,2,3,2,2.5,초기값
김작업,35,48,10,3,4,6,2,4.0,초기값
김작업,50,67,12,5,6,9,3,5.5,초기값
가입전작업자,25,36,8,3,2,4,2,3.0,초기값
가입전작업자,40,55,10,5,5,8,3,4.8,초기값
가입전작업자,60,79,13,6,8,13,4,6.7,초기값
"""
        upload = UploadFile(
            filename="fatigue-training.csv",
            file=io.BytesIO(csv_text.encode("utf-8-sig")),
        )

        result = asyncio.run(api.train_fatigue_dataset(upload, self.admin, self.db))

        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["global_sample_count"], 6)
        self.assertEqual(result["personal_model_count"], 2)
        self.assertEqual(result["ignored_columns"], ["메모"])
        active_models = self.db.query(FatigueModel).filter(
            FatigueModel.organization_id == self.organization.id,
            FatigueModel.is_active.is_(True),
        ).all()
        self.assertEqual(len(active_models), 3)
        self.assertTrue(all(model.source == "dataset_import" for model in active_models))
        own_model = next(model for model in active_models if model.worker_name == "김작업")
        future_model = next(model for model in active_models if model.worker_name == "가입전작업자")
        self.assertEqual(own_model.user_id, self.user.id)
        self.assertIsNone(future_model.user_id)
        self.assertEqual(own_model.actual_response_count, 0)
        self.assertEqual(len(own_model.parameters["coefficients"]), 10)

        prediction = api.predict_fatigue_after_work(
            api.FatiguePredictionRequest(
                schedule_ids=[],
                total_seconds=3000,
                work_seconds=2400,
                driving_seconds=600,
                unknown_seconds=0,
                team_size=2,
            ),
            self.user,
            self.db,
        )
        self.assertIsNotNone(prediction["predicted_borg_cr10"])
        self.assertEqual(prediction["prediction_source"], "stored_initial")
        self.assertEqual(prediction["model_version"], "dataset-recovery-ridge-v3")
        self.assertTrue(prediction["survey_required"])
        self.assertEqual(prediction["cumulative_fatigue"], 0)
        self.assertEqual(prediction["recovery_minutes"], 0)

    def test_recovery_load_halves_and_resets_at_day_boundary(self):
        history = [{
            "worker_name": "김작업",
            "started_at": datetime(2026, 8, 19, 8, 0),
            "completed_at": datetime(2026, 8, 19, 9, 0),
            "features": {"work_minutes": 30},
            "borg_cr10": 4,
        }]
        self.assertEqual(fatigue_decay(4, 120, 120), 2)
        self.assertEqual(
            recovered_daily_load(history, datetime(2026, 8, 19, 11, 0), 120),
            2,
        )
        self.assertEqual(
            recovery_minutes_since_previous(history, datetime(2026, 8, 19, 11, 0)),
            120,
        )
        self.assertEqual(
            recovered_daily_load(history, datetime(2026, 8, 20, 9, 0), 120),
            0,
        )
        self.assertEqual(
            recovery_minutes_since_previous(history, datetime(2026, 8, 20, 9, 0)),
            0,
        )

    def test_fixed_worker_recovery_order_and_first_dispatch_rules(self):
        self.assertEqual(WORKER_RECOVERY_HALF_LIFE_MINUTES, {
            "김경언": 60.0,
            "최현": 90.0,
            "강경래": 120.0,
            "정하람": 180.0,
            "박우민": 360.0,
        })
        first_dispatch = with_recovery_features(
            {"work_minutes": 30},
            [],
            datetime(2026, 8, 19, 9, 0),
            WORKER_RECOVERY_HALF_LIFE_MINUTES["강경래"],
        )
        self.assertEqual(first_dispatch["cumulative_fatigue"], 0)
        self.assertEqual(first_dispatch["recovery_minutes"], 0)

        prior = [{
            "worker_name": "강경래",
            "started_at": datetime(2026, 8, 19, 9, 0),
            "completed_at": datetime(2026, 8, 19, 9, 30),
            "features": {"work_minutes": 30},
            "borg_cr10": 4,
        }]
        second_dispatch = with_recovery_features(
            {"work_minutes": 50},
            prior,
            datetime(2026, 8, 19, 10, 0),
            WORKER_RECOVERY_HALF_LIFE_MINUTES["강경래"],
        )
        self.assertEqual(second_dispatch["cumulative_fatigue"], 3.3636)
        self.assertEqual(second_dispatch["recovery_minutes"], 30)

        remaining = {
            name: fatigue_decay(4, 60, half_life)
            for name, half_life in WORKER_RECOVERY_HALF_LIFE_MINUTES.items()
        }
        self.assertLess(remaining["김경언"], remaining["최현"])
        self.assertLess(remaining["최현"], remaining["강경래"])
        self.assertLess(remaining["강경래"], remaining["정하람"])
        self.assertLess(remaining["정하람"], remaining["박우민"])

    def test_easypickup_workbook_creates_five_linked_personal_models(self):
        dataset_path = (
            Path(__file__).resolve().parents[1]
            / "datas"
            / "이지픽업_CR10_모델입력_현재최종.xlsx"
        )
        parsed = parse_fatigue_dataset(dataset_path.name, dataset_path.read_bytes())
        self.assertEqual(parsed["dataset_format"], "easypickup_cr10_dispatch")
        self.assertEqual(parsed["source_row_count"], 178)
        self.assertEqual(parsed["row_count"], 118)
        self.assertEqual(
            parsed["derived_feature_validation"],
            {"provided_rows": 0, "validated_rows": 0},
        )
        self.assertEqual(
            Counter(sample["worker_name"] for sample in parsed["samples"]),
            Counter({"강경래": 23, "김경언": 24, "박우민": 24, "정하람": 24, "최현": 23}),
        )

        result = provision_personalized_workers(
            self.db,
            self.organization,
            parsed["samples"],
        )
        self.assertEqual(len(result["accounts"]), 5)
        self.assertTrue(all(item["status"] == "created" for item in result["accounts"]))
        self.assertEqual(result["training"]["personal_model_count"], 5)
        active_personal = self.db.query(FatigueModel).filter(
            FatigueModel.organization_id == self.organization.id,
            FatigueModel.scope == "personal",
            FatigueModel.is_active.is_(True),
        ).all()
        self.assertEqual(len(active_personal), 5)
        self.assertTrue(all(model.user_id is not None for model in active_personal))
        recovery_by_name = {
            model.worker_name: model.parameters["recovery"]["half_life_minutes"]
            for model in active_personal
        }
        self.assertEqual(recovery_by_name, {
            "강경래": 120.0,
            "김경언": 60.0,
            "박우민": 360.0,
            "정하람": 180.0,
            "최현": 90.0,
        })
        choi_followup = next(
            sample for sample in parsed["samples"]
            if sample["worker_name"] == "최현" and sample["dispatch_id"] == "6"
        )
        self.assertEqual(choi_followup["features"]["cumulative_fatigue"], 2.778)
        self.assertEqual(choi_followup["features"]["recovery_minutes"], 30)
        provenance_by_name = {
            model.worker_name: model.parameters["training_data_provenance"]
            for model in active_personal
        }
        self.assertEqual(provenance_by_name["최현"], {"recorded_response": 23})
        self.assertEqual(provenance_by_name["김경언"], {"recorded_response": 24})
        self.assertEqual(provenance_by_name["강경래"], {"synthetic_assumption": 23})
        self.assertEqual(provenance_by_name["박우민"], {"synthetic_assumption": 24})
        self.assertEqual(provenance_by_name["정하람"], {"synthetic_assumption": 24})

    def test_personalized_worker_second_dispatch_end_to_end(self):
        dataset_path = (
            Path(__file__).resolve().parents[1]
            / "datas"
            / "이지픽업_CR10_모델입력_현재최종.xlsx"
        )
        parsed = parse_fatigue_dataset(dataset_path.name, dataset_path.read_bytes())
        provision_personalized_workers(self.db, self.organization, parsed["samples"])
        worker = self.db.query(User).filter(
            User.organization_id == self.organization.id,
            User.full_name == "최현",
        ).one()

        first_schedule = Schedule(
            organization_id=self.organization.id,
            신청번호="E2E-1",
            출동일시=datetime(2026, 8, 19, 9, 0),
            품명="책상",
            설치장소="공학관 101호",
            필요인원수=1,
            투입인원수=1,
            가용명단="최현, 김경언",
        )
        second_schedule = Schedule(
            organization_id=self.organization.id,
            신청번호="E2E-2",
            출동일시=datetime(2026, 8, 19, 10, 0),
            품명="의자",
            설치장소="공학관 102호",
            필요인원수=1,
            투입인원수=1,
            가용명단="최현, 김경언",
        )
        self.db.add_all([first_schedule, second_schedule])
        self.db.flush()
        self.db.add_all([
            StaffingDecision(
                organization_id=self.organization.id,
                confirmed_by_user_id=self.admin.id,
                dispatch_time=first_schedule.출동일시,
                schedule_ids=[first_schedule.id],
                selected_workers=[worker.full_name],
                confirmed_at=datetime(2026, 8, 19, 8, 30),
            ),
            StaffingDecision(
                organization_id=self.organization.id,
                confirmed_by_user_id=self.admin.id,
                dispatch_time=second_schedule.출동일시,
                schedule_ids=[second_schedule.id],
                selected_workers=[worker.full_name],
                confirmed_at=datetime(2026, 8, 19, 9, 50),
            ),
        ])
        self.db.commit()

        first = api.create_work_session(
            api.WorkSessionCreate(
                client_session_id="e2e-first",
                worker_name=worker.full_name,
                schedule_ids=[first_schedule.id],
                application_numbers=[first_schedule.신청번호],
                started_at=datetime(2026, 8, 19, 9, 0, tzinfo=api.KST),
                completed_at=datetime(2026, 8, 19, 9, 30, tzinfo=api.KST),
                total_seconds=1800,
                work_seconds=1800,
                borg_cr10=3.5,
            ),
            worker,
            self.db,
        )
        self.assertEqual(first["feature_snapshot"]["cumulative_fatigue"], 0)
        self.assertEqual(first["feature_snapshot"]["recovery_minutes"], 0)

        prediction = api.predict_fatigue_after_work(
            api.FatiguePredictionRequest(
                schedule_ids=[second_schedule.id],
                started_at=datetime(2026, 8, 19, 10, 0, tzinfo=api.KST),
                total_seconds=3000,
                work_seconds=3000,
                team_size=1,
            ),
            worker,
            self.db,
        )
        self.assertEqual(prediction["recovery_half_life_minutes"], 90)
        self.assertEqual(prediction["recovery_minutes"], 30)
        self.assertEqual(prediction["cumulative_fatigue"], 2.778)
        self.assertEqual(prediction["features"]["cumulative_fatigue"], 2.778)
        self.assertIsNotNone(prediction["predicted_borg_cr10"])

        status = next(
            item for item in api.workers_status_today(self.admin, self.db)
            if item["worker_name"] == worker.full_name
        )
        self.assertEqual(status["recovery_half_life_minutes"], 90)
        self.assertEqual(status["cumulative_fatigue"], 2.778)
        proposal = next(
            item for item in api.staffing_recommendations_today(self.admin, self.db)
            if item["dispatch_time"] == second_schedule.출동일시
        )
        worker_detail = next(
            item for item in proposal["worker_details"]
            if item["worker_name"] == worker.full_name
        )
        self.assertIn("누적피로도 2.778", worker_detail["reason"])

    def test_uploaded_dataset_rejects_missing_required_columns_without_mutation(self):
        upload = UploadFile(
            filename="bad.csv",
            file=io.BytesIO("작업자,작업시간,예상피로도\n김작업,20,3".encode("utf-8")),
        )
        with self.assertRaises(api.HTTPException) as raised:
            asyncio.run(api.train_fatigue_dataset(upload, self.admin, self.db))

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("필수 열", raised.exception.detail)
        self.assertEqual(self.db.query(FatigueModel).count(), 0)


if __name__ == "__main__":
    unittest.main()
