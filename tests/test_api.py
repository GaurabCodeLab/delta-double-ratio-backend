from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.models import OptionLeg, Quote


class FakeExchange:
    def __init__(self, *args, **kwargs) -> None:
        self.positions: list[OptionLeg] = []

    async def get_index_price(self, symbol: str) -> float:
        return 100.0

    async def parse_option_positions(self):
        return self.positions

    async def get_best_quote(self, product_id: int):
        return Quote(best_bid=100.0, best_ask=101.0)


class FakeExecutor:
    async def execute_market_single_submission_with_fill_confirmation(self, **kwargs):
        return 1.0


@pytest.fixture
def api_module(monkeypatch):
    monkeypatch.setenv("DELTA_API_KEY", "key")
    monkeypatch.setenv("DELTA_API_SECRET", "secret")
    monkeypatch.setenv("DELTA_BASE_URL", "https://example.com")
    monkeypatch.setenv("DELTA_SSL_VERIFY", "false")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("PNL_PROFIT_THRESHOLD_USD", "10")
    monkeypatch.setenv("PNL_LOSS_THRESHOLD_USD", "-20")
    monkeypatch.setenv("RATIO_SPREADS", "1:2")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    import src.exchange_client as exchange_client_module
    exchange_client_module.DeltaExchangeClient = FakeExchange

    sys.modules.pop("src.api", None)
    api_module = importlib.import_module("src.api")
    return api_module


def test_health_endpoint(api_module) -> None:
    client = TestClient(api_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_start_and_status_endpoints(api_module) -> None:
    api_module.manager.strategy.run = AsyncMock()

    client = TestClient(api_module.app)
    start_resp = client.post("/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "started"

    api_module.manager._task = MagicMock()
    api_module.manager._task.done.return_value = False

    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["running"] is True
    assert status_resp.json()["strategy_state"]["profit_exit_threshold_usd"] == 10.0
    assert status_resp.json()["strategy_state"]["loss_exit_threshold_usd"] == -20.0
    assert status_resp.json()["strategy_state"]["allowed_ratio_spreads"] == [2.0]
    assert status_resp.json()["strategy_state"]["realized_pnl_total"] == 0.0
    assert status_resp.json()["strategy_state"]["unrealized_pnl"] == 0.0
    assert status_resp.json()["strategy_state"]["combined_pnl"] == 0.0


def test_summary_endpoint_returns_strategy_snapshot(api_module) -> None:
    api_module.manager.strategy.exchange = FakeExchange()
    api_module.manager.strategy.state.status_message = "running"
    api_module.manager.strategy.state.action = "monitoring ratio-spreads"
    api_module.manager.strategy.state.status = "monitoring pnl thresholds"

    client = TestClient(api_module.app)
    resp = client.get("/summary")
    assert resp.status_code == 200
    assert "unrealized_pnl" in resp.json()
    assert "positions" in resp.json()
    assert resp.json()["profit_exit_threshold_usd"] == 10.0
    assert resp.json()["loss_exit_threshold_usd"] == -20.0
    assert resp.json()["allowed_ratio_spreads"] == [2.0]
    assert resp.json()["strategy_realized_pnl"] == 0.0
    assert resp.json()["realized_pnl"] == 0.0
    assert resp.json()["combined_pnl"] == resp.json()["realized_pnl"] + resp.json()["unrealized_pnl"]
    assert resp.json()["closed_positions"] == []


def test_summary_includes_closed_position_details(api_module) -> None:
    api_module.manager.strategy.positions._closed_positions = [{
        "fill_id": "fill-closed-1",
        "product_id": 99,
        "symbol": "BTC-TEST-CALL",
        "option_type": "call",
        "strike": 100.0,
        "expiry": "2026-12-31T00:00:00+00:00",
        "position_side": "long",
        "closing_side": "sell",
        "quantity": 1.0,
        "entry_price": 96.0,
        "exit_price": 100.0,
        "contract_value": 1.0,
        "realized_pnl": 4.0,
        "closed_at": "2026-10-01T00:00:00+00:00",
    }]

    with TestClient(api_module.app) as client:
        response = client.get("/summary")

    assert response.status_code == 200
    assert response.json()["closed_positions"][0]["fill_id"] == "fill-closed-1"
    assert response.json()["closed_positions"][0]["realized_pnl"] == 4.0


def test_summary_combined_pnl_adds_signed_values(api_module) -> None:
    exchange = FakeExchange()
    exchange.positions = [
        OptionLeg(
            product_id=99,
            symbol="BTC-TEST-CALL",
            option_type="call",
            strike=100.0,
            expiry=datetime(2026, 1, 1, tzinfo=timezone.utc),
            size=1.0,
            entry_price=96.0,
        )
    ]
    api_module.manager.strategy.exchange = exchange
    api_module.manager.strategy.state.tracked_product_ids = {99}
    api_module.manager.strategy.state.realized_pnl_total = -8.0

    with TestClient(api_module.app) as client:
        response = client.get("/summary")

    payload = response.json()
    assert payload["unrealized_pnl"] == pytest.approx(4.0)
    assert payload["realized_pnl"] == pytest.approx(-8.0)
    assert payload["combined_pnl"] == pytest.approx(-4.0)
    assert payload["combined_pnl"] == pytest.approx(payload["realized_pnl"] + payload["unrealized_pnl"])


def test_logs_endpoint_returns_logs_after_startup(api_module) -> None:
    logger = logging.getLogger("api.logs.test")

    with TestClient(api_module.app) as client:
        logger.info("log endpoint test message")
        response = client.get("/logs")

    assert response.status_code == 200
    assert response.json()["count"] >= 1
    assert any(
        item["message"] == "log endpoint test message"
        for item in response.json()["logs"]
    )
