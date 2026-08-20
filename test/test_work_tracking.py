import unittest
from datetime import date, datetime

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
from models import Base, FatigueModel, Organization, Schedule, StaffingDecision, User, WorkSession  # noqa: E402


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
        self.assertEqual(status["daily_load"], 120.0)

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
        self.assertEqual(len(model.parameters["coefficients"]), 8)


if __name__ == "__main__":
    unittest.main()
