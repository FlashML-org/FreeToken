"""HTTP probe contracts: liveness, admission readiness, and legacy desktop health."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.server.control_api import register_control_routes


def state(phase):
    return SimpleNamespace(
        maintenance_state=phase, fatal_error=None, instance_id="instance",
        config=SimpleNamespace(served_model_name="model"), ready_at=None,
        load_progress=SimpleNamespace(phase="weights", done_bytes=10, total_bytes=100),
    )


@pytest.mark.parametrize("phase", ["loading", "rebuilding", "failed", "stopping", "serving"])
def test_only_serving_is_ready_and_health_remains_compatible(phase):
    current = state(phase)
    app = FastAPI(version="test-version")
    register_control_routes(app, lambda: current)
    expected = ({
        "status": "loading", "phase": "weights",
        "progress": {"done_bytes": 10, "total_bytes": 100},
        "model": "model", "instance_id": "instance",
    } if phase == "loading" else {
        "status": "ok", "model": "model", "instance_id": "instance",
        "uptime_s": 0, "maintenance": phase, "version": "test-version",
    })
    with TestClient(app) as http:
        legacy = http.get("/health")
        ready = http.get("/readyz")
        live = http.get("/healthz")
    assert legacy.status_code == 200
    assert legacy.json() == expected
    assert ready.status_code == (200 if phase == "serving" else 503)
    assert ready.json() == expected
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}


def test_liveness_does_not_access_engine_state():
    def unavailable():
        raise AssertionError("engine state must not be read by a liveness probe")

    app = FastAPI()
    register_control_routes(app, unavailable)
    with TestClient(app) as http:
        assert http.get("/healthz").status_code == 200


def test_worker_death_wins_over_a_racing_serving_state():
    current = state("serving")
    current.fatal_error = "scheduler exited"
    app = FastAPI()
    register_control_routes(app, lambda: current)
    with TestClient(app) as http:
        assert http.get("/health").status_code == 200
        response = http.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {
        "status": "error", "message": "scheduler exited", "instance_id": "instance",
    }


def test_lifecycle_changes_are_observed_without_restarting_http():
    current = state("loading")
    app = FastAPI()
    register_control_routes(app, lambda: current)
    with TestClient(app) as http:
        assert http.get("/readyz").status_code == 503
        current.maintenance_state = "serving"
        assert http.get("/readyz").status_code == 200
        current.maintenance_state = "stopping"
        assert http.get("/readyz").status_code == 503
