from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from src.exchange_client import DeltaExchangeClient
from src.models import OptionLeg, RatioSpread


class PositionManager:
    def __init__(self, exchange: DeltaExchangeClient) -> None:
        self.exchange = exchange
        self._realized_tracking_started_at = datetime.now(timezone.utc)
        self._seen_fill_ids: set[str] = set()
        self._closed_positions: list[dict[str, Any]] = []

    @property
    def closed_positions(self) -> list[dict[str, Any]]:
        return [dict(position) for position in self._closed_positions]

    async def detect_ratio_spreads(self, accepted_ratios: list[float]) -> list[RatioSpread]:
        """
        Detect all call and put ratio spreads simultaneously.
        Returns a list of RatioSpread objects (both call and put, if they exist).
        """
        legs = await self.exchange.parse_option_positions()
        if not legs:
            raise RuntimeError("No open option positions found")

        if not accepted_ratios:
            raise ValueError("At least one ratio must be configured for ratio spread detection.")

        # Separate by option type
        call_legs = [x for x in legs if x.option_type == "call"]
        put_legs = [x for x in legs if x.option_type == "put"]

        found_spreads: list[RatioSpread] = []

        # Detect call ratio spreads
        if call_legs:
            call_ratio = self._find_ratio_spread_for_type(call_legs, accepted_ratios)
            if call_ratio is not None:
                found_spreads.append(call_ratio)

        # Detect put ratio spreads
        if put_legs:
            put_ratio = self._find_ratio_spread_for_type(put_legs, accepted_ratios)
            if put_ratio is not None:
                found_spreads.append(put_ratio)

        if not found_spreads:
            leg_summary = ", ".join(
                f"pid={x.product_id}|{x.option_type}|strike={x.strike}|exp={x.expiry.isoformat()}|size={x.size}"
                for x in legs
            )
            expected_summary = ", ".join(str(r) for r in accepted_ratios)
            raise RuntimeError(
                "No valid ratio spread found for ratios: "
                f"{expected_summary}. Parsed legs: "
                + leg_summary
            )

        return found_spreads

    def _find_ratio_spread_for_type(self, legs: list[OptionLeg], accepted_ratios: list[float]) -> RatioSpread | None:
        """
        Find a single ratio spread within legs of the same option type.
        Returns the first matching ratio spread or None if none found.
        """
        longs = [x for x in legs if x.size > 0]
        shorts = [x for x in legs if x.size < 0]

        for long_leg in longs:
            for short_leg in shorts:
                same_expiry = (
                    long_leg.expiry == short_leg.expiry
                    or long_leg.expiry.date() == short_leg.expiry.date()
                )
                if not same_expiry:
                    continue

                long_qty = abs(long_leg.size)
                short_qty = abs(short_leg.size)
                if long_qty <= 0:
                    continue

                actual_ratio = short_qty / long_qty
                for expected_ratio in accepted_ratios:
                    if math.isclose(
                        actual_ratio,
                        expected_ratio,
                        rel_tol=2e-2,
                        abs_tol=1e-8,
                    ):
                        return RatioSpread(
                            long_leg=long_leg,
                            short_leg=short_leg,
                            short_leg_each_qty=long_qty,
                            ratio=actual_ratio,
                        )

        return None

    async def compute_unrealized_pnl(self, ratio: RatioSpread) -> float:
        # Calculate PnL based on entry price and actual liquidation prices.
        # Long positions close at best_bid (conservative: worst price when selling)
        # Short positions close at best_ask (conservative: worst price when buying back)
        long_q = await self.exchange.get_best_quote(ratio.long_leg.product_id)
        short_q = await self.exchange.get_best_quote(ratio.short_leg.product_id)

        long_qty = abs(ratio.long_leg.size)
        short_qty = abs(ratio.short_leg.size)

        # Long leg PnL: (best_bid - entry_price) * qty * contract_value
        # (position closed by selling at bid price)
        long_leg_pnl = (
            long_qty
            * ratio.long_leg.contract_value
            * (long_q.best_bid - ratio.long_leg.entry_price)
        )

        # Short leg PnL: (entry_price - best_ask) * qty * contract_value
        # (position closed by buying back at ask price)
        short_leg_pnl = (
            short_qty
            * ratio.short_leg.contract_value
            * (ratio.short_leg.entry_price - short_q.best_ask)
        )

        return long_leg_pnl + short_leg_pnl

    async def compute_unrealized_leg_pnl(self, leg: OptionLeg) -> float:
        quote = await self.exchange.get_best_quote(leg.product_id)
        quantity = abs(leg.size)
        if leg.size > 0:
            return quantity * leg.contract_value * (quote.best_bid - leg.entry_price)
        return quantity * leg.contract_value * (leg.entry_price - quote.best_ask)

    async def reconcile_realized_pnl(
        self,
        previous_open_legs: list[OptionLeg],
        current_open_legs: list[OptionLeg] | None = None,
    ) -> float:
        """Compute gross realized PnL for observed position reductions using new fills."""
        if current_open_legs is None:
            current_open_legs = await self.exchange.parse_option_positions()

        current_by_id = {leg.product_id: leg for leg in current_open_legs}
        realized_total = 0.0
        pending_fill_ids: set[str] = set()
        pending_closed_positions: list[dict[str, Any]] = []

        for leg in previous_open_legs:
            current_leg = current_by_id.get(leg.product_id)
            if current_leg is None or leg.size * current_leg.size <= 0:
                closed_quantity = abs(leg.size)
            else:
                closed_quantity = max(0.0, abs(leg.size) - abs(current_leg.size))
            if closed_quantity <= 1e-12:
                continue

            closing_side = "sell" if leg.size > 0 else "buy"
            fills = await self.exchange.get_recent_fills(leg.product_id)
            matching_fills = sorted(
                (fill for fill in fills if self._is_new_closing_fill(fill, leg.product_id, closing_side)),
                key=self._fill_timestamp,
            )
            remaining_quantity = closed_quantity
            leg_realized = 0.0
            leg_fill_ids: set[str] = set()

            for fill in matching_fills:
                fill_id = self._fill_id(fill)
                if fill_id is None:
                    raise RuntimeError(f"Closing fill for product {leg.product_id} has no stable fill ID")
                if fill_id in self._seen_fill_ids or fill_id in pending_fill_ids:
                    continue

                fill_quantity = self._fill_number(fill, "size", "fill_size", "quantity")
                fill_price = self._fill_number(fill, "price", "fill_price")
                quantity = min(remaining_quantity, fill_quantity)
                if quantity <= 0:
                    continue

                if leg.size > 0:
                    fill_realized = (fill_price - leg.entry_price) * quantity * leg.contract_value
                else:
                    fill_realized = (leg.entry_price - fill_price) * quantity * leg.contract_value
                leg_realized += fill_realized
                remaining_quantity -= quantity
                leg_fill_ids.add(fill_id)
                closed_at = self._fill_timestamp(fill)
                if closed_at is None:
                    raise RuntimeError(f"Closing fill {fill_id} for product {leg.product_id} has no valid timestamp")
                pending_closed_positions.append(
                    {
                        "fill_id": fill_id,
                        "product_id": leg.product_id,
                        "symbol": leg.symbol,
                        "type": leg.option_type,
                        "option_type": leg.option_type,
                        "strike": leg.strike,
                        "expiry": leg.expiry.isoformat(),
                        "side": "Buy" if leg.size > 0 else "Sell",
                        "position_side": "long" if leg.size > 0 else "short",
                        "closing_side": closing_side,
                        "size": quantity if leg.size > 0 else -quantity,
                        "quantity": quantity,
                        "entry": leg.entry_price,
                        "entry_price": leg.entry_price,
                        "closed_price": fill_price,
                        "exit_price": fill_price,
                        "contract_value": leg.contract_value,
                        "realized_pnl": fill_realized,
                        "closed_time": closed_at.isoformat(),
                        "closed_at": closed_at.isoformat(),
                    }
                )
                if remaining_quantity <= 1e-12:
                    break

            if remaining_quantity > 1e-8:
                raise RuntimeError(
                    f"Fills cover only {closed_quantity - remaining_quantity:g} of "
                    f"{closed_quantity:g} closed contracts for product {leg.product_id}; "
                    "realized PnL was not booked"
                )
            realized_total += leg_realized
            pending_fill_ids.update(leg_fill_ids)

        self._seen_fill_ids.update(pending_fill_ids)
        self._closed_positions.extend(pending_closed_positions)
        return realized_total

    def _is_new_closing_fill(self, fill: dict[str, Any], product_id: int, side: str) -> bool:
        try:
            fill_product_id = int(fill.get("product_id"))
        except (TypeError, ValueError):
            return False
        fill_side = str(fill.get("side") or "").lower()
        timestamp = self._fill_timestamp(fill)
        return (
            fill_product_id == product_id
            and fill_side == side
            and timestamp is not None
            and timestamp >= self._realized_tracking_started_at
        )

    @staticmethod
    def _fill_id(fill: dict[str, Any]) -> str | None:
        for key in ("id", "fill_id", "trade_id"):
            value = fill.get(key)
            if value is not None and str(value):
                return str(value)
        return None

    @staticmethod
    def _fill_number(fill: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = fill.get(key)
            if value is not None:
                number = float(value)
                if math.isfinite(number) and number > 0:
                    return number
        raise RuntimeError(f"Fill is missing a positive numeric value for {', '.join(keys)}")

    @staticmethod
    def _fill_timestamp(fill: dict[str, Any]) -> datetime | None:
        value = next(
            (fill.get(key) for key in ("created_at", "created_time", "timestamp", "time") if fill.get(key) is not None),
            None,
        )
        if isinstance(value, (int, float)):
            timestamp = float(value)
            if abs(timestamp) >= 1e12:
                timestamp /= 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            except ValueError:
                return None
        return None

