import os
import unittest
from datetime import datetime

# auth 모듈을 import하기 전에 로컬 테스트에서만 재설정 토큰 반환을 허용한다.
os.environ["PASSWORD_RESET_RETURN_TOKEN"] = "true"
os.environ["AUTH_SECRET_KEY"] = "test-secret-that-is-long-enough-for-local-tests"

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from auth import router
from db import Base, get_db
from models import FatigueModel, Organization, User


class AuthApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        cls.Session = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        Base.metadata.create_all(bind=cls.engine)

        app = FastAPI()
        app.include_router(router)

        def test_db():
            db = cls.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = test_db
        cls.client = TestClient(app)

    def setUp(self):
        with self.Session() as db:
            db.query(FatigueModel).delete()
            db.query(User).delete()
            db.query(Organization).delete()
            db.commit()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.engine.dispose()

    def _signup(self):
        return self.client.post(
            "/auth/signup",
            json={
                "username": "testuser",
                "password": "password123",
                "full_name": "홍길동",
                "email": "USER@example.com",
                "organization_mode": "create",
                "organization_name": "테스트 관재팀",
            },
        )

    def test_signup_hashes_password_and_rejects_duplicates(self):
        response = self._signup()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["username"], "testuser")
        self.assertEqual(response.json()["email"], "user@example.com")
        self.assertNotIn("password", response.json())
        self.assertEqual(response.json()["role"], "admin")
        self.assertEqual(response.json()["organization_name"], "테스트 관재팀")
        self.assertEqual(len(response.json()["organization_entry_code"]), 6)

        with self.Session() as db:
            user = db.query(User).filter(User.username == "testuser").one()
            self.assertTrue(user.password_hash.startswith("scrypt$"))
            self.assertNotIn("password123", user.password_hash)

        duplicate = self._signup()
        self.assertEqual(duplicate.status_code, 409)

    def test_login_and_find_id(self):
        self._signup()

        failed = self.client.post(
            "/auth/login",
            json={"username": "testuser", "password": "wrong-password1"},
        )
        self.assertEqual(failed.status_code, 401)

        logged_in = self.client.post(
            "/auth/login",
            json={"username": "TESTUSER", "password": "password123"},
        )
        self.assertEqual(logged_in.status_code, 200)
        self.assertEqual(logged_in.json()["token_type"], "bearer")
        self.assertTrue(logged_in.json()["access_token"].startswith("v1."))
        token = logged_in.json()["access_token"]

        unauthenticated = self.client.get("/auth/me")
        self.assertEqual(unauthenticated.status_code, 401)
        current_user = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(current_user.status_code, 200)
        self.assertEqual(current_user.json()["full_name"], "홍길동")
        self.assertEqual(current_user.json()["role"], "admin")
        tampered = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}broken"})
        self.assertEqual(tampered.status_code, 401)

        found = self.client.post(
            "/auth/find-id",
            json={"full_name": "홍길동", "email": "user@example.com"},
        )
        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.json()["username"], "tes*****")

    def test_password_reset_token_is_single_use(self):
        self._signup()

        requested = self.client.post(
            "/auth/forgot-password",
            json={"email": "user@example.com"},
        )
        self.assertEqual(requested.status_code, 200)
        token = requested.json()["reset_token"]

        with self.Session() as db:
            user = db.query(User).filter(User.username == "testuser").one()
            self.assertNotEqual(user.reset_token_hash, token)
            user.reset_token_expires_at = datetime(2000, 1, 1)
            db.commit()

        expired = self.client.post(
            "/auth/reset-password",
            json={"token": token, "new_password": "new-password456"},
        )
        self.assertEqual(expired.status_code, 400)

        requested = self.client.post(
            "/auth/forgot-password",
            json={"email": "user@example.com"},
        )
        token = requested.json()["reset_token"]

        reset = self.client.post(
            "/auth/reset-password",
            json={"token": token, "new_password": "new-password456"},
        )
        self.assertEqual(reset.status_code, 200)

        reused = self.client.post(
            "/auth/reset-password",
            json={"token": token, "new_password": "another-password789"},
        )
        self.assertEqual(reused.status_code, 400)

        old_login = self.client.post(
            "/auth/login",
            json={"username": "testuser", "password": "password123"},
        )
        new_login = self.client.post(
            "/auth/login",
            json={"username": "testuser", "password": "new-password456"},
        )
        self.assertEqual(old_login.status_code, 401)
        self.assertEqual(new_login.status_code, 200)

    def test_password_and_username_validation(self):
        response = self.client.post(
            "/auth/signup",
            json={
                "username": "bad id",
                "password": "short",
                "full_name": "홍길동",
                "email": "not-an-email",
                "organization_mode": "create",
                "organization_name": "테스트 조직",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_worker_joins_organization_with_entry_code(self):
        admin = self._signup()
        code = admin.json()["organization_entry_code"]
        worker = self.client.post(
            "/auth/signup",
            json={
                "username": "worker01",
                "password": "password123",
                "full_name": "김작업",
                "email": "worker@example.com",
                "organization_mode": "join",
                "entry_code": code,
            },
        )
        self.assertEqual(worker.status_code, 201)
        self.assertEqual(worker.json()["role"], "worker")
        self.assertEqual(worker.json()["organization_id"], admin.json()["organization_id"])
        self.assertIsNone(worker.json()["organization_entry_code"])

        invalid = self.client.post(
            "/auth/signup",
            json={
                "username": "worker02",
                "password": "password123",
                "full_name": "이작업",
                "email": "worker2@example.com",
                "organization_mode": "join",
                "entry_code": "ZZZZZZ",
            },
        )
        self.assertEqual(invalid.status_code, 404)


if __name__ == "__main__":
    unittest.main()
