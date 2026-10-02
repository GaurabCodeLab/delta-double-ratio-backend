from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from src.models import OptionLeg, Quote, RatioSpread
from src.position_manager import PositionManager


class FakeExchange:
    def __init__(self, quotes: dict[int, Quote]) -> None:
        self._quotes = quotes
        self.parsed_positions = []
        self.fills: dict[int, list[dict]] = {}

    async def parse_option_positions(self):
        return self.parsed_positions

    async def get_best_quote(self, product_id: int) -> Quote:
        return self._quotes[product_id]

    async def get_recent_fills(self, product_id: int | None = None, limit: int = 50):
        if product_id is None:
            all_fills = []
            for fills in self.fills.values():
                all_fills.extend(fills)
            return all_fills
        return self.fills.get(product_id, [])


@pytest.mark.asyncio
async def test_detect_ratio_spreads_returns_call_and_put() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    call_long = OptionLeg(1, "BTC-1C", "call", 100.0, now, 1.0, 101.0)
    call_short = OptionLeg(2, "BTC-2C", "call", 100.0, now, -2.0, 102.0)
    put_long = OptionLeg(3, "BTC-1P", "put", 95.0, now, 1.0, 96.0)
    put_short = OptionLeg(4, "BTC-2P", "put", 95.0, now, -2.0, 97.0)

    fake_exchange = FakeExchange(
        quotes={
            1: Quote(best_bid=101.5, best_ask=102.5),
            2: Quote(best_bid=100.5, best_ask=101.5),
            3: Quote(best_bid=96.5, best_ask=97.5),
            4: Quote(best_bid=94.5, best_ask=95.5),
        }
    )
    fake_exchange.parsed_positions = [call_long, call_short, put_long, put_short]

    manager = PositionManager(fake_exchange)
    spreads = await manager.detect_ratio_spreads([2.0])

    assert len(spreads) == 2
    assert {spread.long_leg.option_type for spread in spreads} == {"call", "put"}
    assert all(spread.ratio == 2.0 for spread in spreads)


@pytest.mark.asyncio
async def test_compute_unrealized_pnl_uses_best_bid_and_ask() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    long_leg = OptionLeg(10, "BTC-LONG", "call", 100.0, now, 1.0, 95.0)
    short_leg = OptionLeg(11, "BTC-SHORT", "call", 100.0, now, -2.0, 96.0)

    ratio = RatioSpread(
        long_leg=long_leg,
        short_leg=short_leg,
        short_leg_each_qty=1.0,
        ratio=2.0,
    )

    fake_exchange = FakeExchange(
        quotes={
            10: Quote(best_bid=106.0, best_ask=107.0),
            11: Quote(best_bid=98.0, best_ask=99.0),
        }
    )

    manager = PositionManager(fake_exchange)
    pnl = await manager.compute_unrealized_pnl(ratio)

    # Long pnl = (106 - 95) * 1 = 11
    # Short pnl = (96 - 99) * 2 = -6
    assert pnl == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_reconcile_realized_pnl_for_manually_closed_positions() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    long_leg = OptionLeg(10, "BTC-LONG", "call", 100.0, now, 1.0, 95.0)
    short_leg = OptionLeg(11, "BTC-SHORT", "call", 100.0, now, -2.0, 96.0)

    fake_exchange = FakeExchange(quotes={})
    manager = PositionManager(fake_exchange)
    fill_time = datetime.now(timezone.utc).isoformat()
    fake_exchange.fills = {
        10: [{"id": "fill-10", "product_id": 10, "side": "sell", "size": "1", "price": "106.0", "created_at": fill_time}],
        11: [{"id": "fill-11", "product_id": 11, "side": "buy", "size": "2", "price": "99.0", "created_at": fill_time}],
    }
    realized = await manager.reconcile_realized_pnl([long_leg, short_leg], [])

    assert realized == pytest.approx(5.0)
    closed_positions = manager.closed_positions
    assert len(closed_positions) == 2
    assert {item["product_id"] for item in closed_positions} == {10, 11}
    assert sum(item["realized_pnl"] for item in closed_positions) == pytest.approx(realized)
    assert all(item["closed_at"] == fill_time for item in closed_positions)
    long_closed = next(item for item in closed_positions if item["product_id"] == 10)
    assert {
        "symbol": long_closed["symbol"],
        "type": long_closed["type"],
        "strike": long_closed["strike"],
        "expiry": long_closed["expiry"],
        "side": long_closed["side"],
        "size": long_closed["size"],
        "entry": long_closed["entry"],
        "closed_price": long_closed["closed_price"],
        "closed_time": long_closed["closed_time"],
        "realized_pnl": long_closed["realized_pnl"],
    } == {
        "symbol": "BTC-LONG",
        "type": "call",
        "strike": 100.0,
        "expiry": now.isoformat(),
        "side": "Buy",
        "size": 1.0,
        "entry": 95.0,
        "closed_price": 106.0,
        "closed_time": fill_time,
        "realized_pnl": 11.0,
    }


@pytest.mark.asyncio
async def test_reconcile_realized_pnl_handles_partial_closes_without_rebooking() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    previous = OptionLeg(10, "BTC-LONG", "call", 100.0, now, 1.0, 95.0)
    current = OptionLeg(10, "BTC-LONG", "call", 100.0, now, 0.5, 95.0)
    fake_exchange = FakeExchange(quotes={})
    manager = PositionManager(fake_exchange)
    fake_exchange.fills = {
        10: [{
            "id": "partial-fill-1",
            "product_id": 10,
            "side": "sell",
            "size": "0.5",
            "price": "106.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }]
    }

    realized = await manager.reconcile_realized_pnl([previous], [current])
    repeated_snapshot = await manager.reconcile_realized_pnl([current], [current])

    assert realized == pytest.approx(5.5)
    assert repeated_snapshot == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_reconcile_does_not_book_when_fill_history_is_incomplete() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    previous = OptionLeg(10, "BTC-LONG", "call", 100.0, now, 1.0, 95.0)
    fake_exchange = FakeExchange(quotes={})
    manager = PositionManager(fake_exchange)
    fake_exchange.fills = {
        10: [{
            "id": "partial-fill-1",
            "product_id": 10,
            "side": "sell",
            "size": "0.5",
            "price": "106.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }]
    }

    with pytest.raises(RuntimeError, match="realized PnL was not booked"):
        await manager.reconcile_realized_pnl([previous], [])
    assert manager.closed_positions == []
