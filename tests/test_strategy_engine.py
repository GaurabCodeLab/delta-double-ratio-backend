from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.models import OptionLeg, Quote, RatioSpread
from src.strategy_engine import StrategyEngine


class FakeExchange:
    def __init__(self) -> None:
        self.index_price = 100.0
        self.open_legs: list[OptionLeg] = []
        self.quote_map = {
            1: Quote(best_bid=101.0, best_ask=102.0),
            2: Quote(best_bid=100.5, best_ask=101.5),
            3: Quote(best_bid=102.0, best_ask=103.0),
            4: Quote(best_bid=101.5, best_ask=102.5),
        }

    async def get_index_price(self, symbol: str) -> float:
        return self.index_price

    async def get_best_quote(self, product_id: int) -> Quote:
        return self.quote_map[product_id]

    async def parse_option_positions(self) -> list[OptionLeg]:
        return self.open_legs


class FakePositions:
    def __init__(self, spreads: list[RatioSpread]) -> None:
        self.spreads = spreads
        self.detect_calls = 0

    async def detect_ratio_spreads(self, accepted_ratios: list[float]) -> list[RatioSpread]:
        self.detect_calls += 1
        return self.spreads

    async def compute_unrealized_pnl(self, ratio: RatioSpread) -> float:
        if ratio.long_leg.option_type == "call":
            return 12.0
        return 3.0

    async def reconcile_realized_pnl(
        self,
        previous_open_legs: list[OptionLeg],
        current_open_legs: list[OptionLeg],
    ) -> float:
        return 0.0


class FakeExecutor:
    def __init__(self) -> None:
        self.closed: list[tuple[int, str, float, bool]] = []

    async def execute_market_single_submission_with_fill_confirmation(
        self,
        product_id: int,
        side: str,
        size: float,
        reduce_only: bool,
    ) -> float:
        self.closed.append((product_id, side, size, reduce_only))
        return float(size)


class FailingThenWorkingPositions(FakePositions):
    def __init__(self, spreads: list[RatioSpread]) -> None:
        super().__init__(spreads)
        self._verify_failures = 1

    async def detect_ratio_spreads(self, accepted_ratios: list[float]) -> list[RatioSpread]:
        self.detect_calls += 1
        if self.detect_calls == 2 and self._verify_failures > 0:
            self._verify_failures -= 1
            raise RuntimeError("verify failed")
        return self.spreads


class ErrorThenValidPositions(FakePositions):
    def __init__(self, spreads: list[RatioSpread]) -> None:
        super().__init__(spreads)
        self._pnl_failures = 1

    async def compute_unrealized_pnl(self, ratio: RatioSpread) -> float:
        if self._pnl_failures > 0:
            self._pnl_failures -= 1
            raise RuntimeError("pnl failed")
        return 12.0 if ratio.long_leg.option_type == "call" else 3.0


class RemovalPositions(FakePositions):
    def __init__(self, spreads: list[RatioSpread]) -> None:
        super().__init__(spreads)
        self._call_count = 0

    async def detect_ratio_spreads(self, accepted_ratios: list[float]) -> list[RatioSpread]:
        self.detect_calls += 1
        if self.detect_calls == 2:
            return []
        return self.spreads


class RaisingExchange(FakeExchange):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def get_index_price(self, symbol: str) -> float:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("index unavailable")
        return self.index_price


class FailingExecutor(FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.failures = 1

    async def execute_market_single_submission_with_fill_confirmation(
        self,
        product_id: int,
        side: str,
        size: float,
        reduce_only: bool,
    ) -> float:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("order failed")
        return await super().execute_market_single_submission_with_fill_confirmation(
            product_id, side, size, reduce_only
        )


@pytest.mark.asyncio
async def test_strategy_engine_closes_all_positions_when_total_pnl_hits_threshold() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    call_ratio = RatioSpread(
        long_leg=OptionLeg(1, "CALL-LONG", "call", 100.0, now, 1.0, 100.0),
        short_leg=OptionLeg(2, "CALL-SHORT", "call", 100.0, now, -2.0, 100.0),
        short_leg_each_qty=1.0,
        ratio=2.0,
    )
    put_ratio = RatioSpread(
        long_leg=OptionLeg(3, "PUT-LONG", "put", 95.0, now, 1.0, 100.0),
        short_leg=OptionLeg(4, "PUT-SHORT", "put", 95.0, now, -2.0, 100.0),
        short_leg_each_qty=1.0,
        ratio=2.0,
    )

    settings = Settings(
        api_key="k",
        api_secret="s",
        base_url="https://example.com",
        poll_interval_seconds=0.0,
        profit_exit_threshold_usd=10.0,
        loss_exit_threshold_usd=-20.0,
        ratio_spreads=[2.0],
        log_level="INFO",
        ssl_verify=False,
    )

    fake_exchange = FakeExchange()
    fake_positions = FakePositions([call_ratio, put_ratio])
    fake_executor = FakeExecutor()

    engine = StrategyEngine(fake_exchange, fake_positions, fake_executor, settings)

    await asyncio.wait_for(engine.run(), timeout=0.5)

    assert len(fake_executor.closed) == 4
    assert engine.state.action == "closing all positions"
    assert engine.state.status == "overall pnl threshold reached"
    assert engine.state.trigger_pnl == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_strategy_engine_recovers_from_ratio_verification_errors() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    call_ratio = RatioSpread(
        long_leg=OptionLeg(1, "CALL-LONG", "call", 100.0, now, 1.0, 100.0),
        short_leg=OptionLeg(2, "CALL-SHORT", "call", 100.0, now, -2.0, 100.0),
        short_leg_each_qty=1.0,
        ratio=2.0,
    )

    settings = Settings(
        api_key="k",
        api_secret="s",
        base_url="https://example.com",
        poll_interval_seconds=0.0,
        profit_exit_threshold_usd=10.0,
        loss_exit_threshold_usd=-20.0,
        ratio_spreads=[2.0],
        log_level="INFO",
        ssl_verify=False,
    )

    fake_exchange = FakeExchange()
    fake_positions = FailingThenWorkingPositions([call_ratio])
    fake_executor = FakeExecutor()

    engine = StrategyEngine(fake_exchange, fake_positions, fake_executor, settings)

    await asyncio.wait_for(engine.run(), timeout=0.5)

    assert len(fake_executor.closed) == 2
    assert engine.state.action == "closing all positions"


@pytest.mark.asyncio
async def test_strategy_engine_continues_when_ratio_monitoring_raises() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    call_ratio = RatioSpread(
        long_leg=OptionLeg(1, "CALL-LONG", "call", 100.0, now, 1.0, 100.0),
        short_leg=OptionLeg(2, "CALL-SHORT", "call", 100.0, now, -2.0, 100.0),
        short_leg_each_qty=1.0,
        ratio=2.0,
    )

    settings = Settings(
        api_key="k",
        api_secret="s",
        base_url="https://example.com",
        poll_interval_seconds=0.0,
        profit_exit_threshold_usd=10.0,
        loss_exit_threshold_usd=-20.0,
        ratio_spreads=[2.0],
        log_level="INFO",
        ssl_verify=False,
    )

    fake_exchange = FakeExchange()
    fake_positions = ErrorThenValidPositions([call_ratio])
    fake_executor = FakeExecutor()

    engine = StrategyEngine(fake_exchange, fake_positions, fake_executor, settings)

    await asyncio.wait_for(engine.run(), timeout=0.5)

    assert len(fake_executor.closed) == 2
    assert engine.state.action == "closing all positions"


@pytest.mark.asyncio
async def test_strategy_engine_handles_initial_scan_failures_and_removed_ratios() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    call_ratio = RatioSpread(
        long_leg=OptionLeg(1, "CALL-LONG", "call", 100.0, now, 1.0, 100.0),
        short_leg=OptionLeg(2, "CALL-SHORT", "call", 100.0, now, -2.0, 100.0),
        short_leg_each_qty=1.0,
        ratio=2.0,
    )

    settings = Settings(
        api_key="k",
        api_secret="s",
        base_url="https://example.com",
        poll_interval_seconds=0.0,
        profit_exit_threshold_usd=10.0,
        loss_exit_threshold_usd=-20.0,
        ratio_spreads=[2.0],
        log_level="INFO",
        ssl_verify=False,
    )

    fake_exchange = RaisingExchange()
    fake_positions = RemovalPositions([call_ratio])
    fake_executor = FakeExecutor()

    engine = StrategyEngine(fake_exchange, fake_positions, fake_executor, settings)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert task.done()


@pytest.mark.asyncio
async def test_strategy_engine_logs_and_continues_when_close_fails() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    call_ratio = RatioSpread(
        long_leg=OptionLeg(1, "CALL-LONG", "call", 100.0, now, 1.0, 100.0),
        short_leg=OptionLeg(2, "CALL-SHORT", "call", 100.0, now, -2.0, 100.0),
        short_leg_each_qty=1.0,
        ratio=2.0,
    )

    settings = Settings(
        api_key="k",
        api_secret="s",
        base_url="https://example.com",
        poll_interval_seconds=0.0,
        profit_exit_threshold_usd=10.0,
        loss_exit_threshold_usd=-20.0,
        ratio_spreads=[2.0],
        log_level="INFO",
        ssl_verify=False,
    )

    fake_exchange = FakeExchange()
    fake_positions = FakePositions([call_ratio])
    fake_executor = FailingExecutor()

    engine = StrategyEngine(fake_exchange, fake_positions, fake_executor, settings)

    await asyncio.wait_for(engine.run(), timeout=0.5)

    assert engine.state.action == "closing all positions"
    assert engine.state.status == "overall pnl threshold reached"
