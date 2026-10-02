from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import load_settings
from src.exchange_client import DeltaExchangeClient
from src.logging_config import setup_logging
from src.order_executor import OrderExecutor
from src.position_manager import PositionManager
from src.strategy_engine import StrategyEngine


class JsonLogHandler(logging.Handler):
    def __init__(self, max_records: int = 1000) -> None:
        super().__init__(level=logging.INFO)
        self.records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._records_lock = Lock()
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()

        timestamp = None
        if self.formatter is not None:
            try:
                timestamp = self.formatter.formatTime(record)
            except Exception:
                timestamp = None

        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        if message.startswith("{") and message.endswith("}"):
            try:
                payload["data"] = json.loads(message)
            except Exception:
                pass

        with self._records_lock:
            self.records.append(payload)

    def latest(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._records_lock:
            records = list(self.records)
        if limit is None:
            return records
        return records[-limit:]


class BotManager:
    def __init__(self, strategy: StrategyEngine) -> None:
        self.strategy = strategy
        self._task: asyncio.Task | None = None
        self._lock = Lock()

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._task is not None and not self._task.done():
                return {"status": "already_running", "running": True}
            self._task = asyncio.create_task(self.strategy.run())
            return {"status": "started", "running": True}

    async def collect_summary(self) -> dict[str, Any]:
        exchange = self.strategy.exchange
        logger = logging.getLogger(__name__)
        
        try:
            index_price = await exchange.get_index_price("BTCUSDT")
            logger.info(f"Index price fetched: {index_price}")
        except Exception as e:
            logger.error(f"Failed to fetch index price: {str(e)}")
            raise
        
        try:
            legs = await exchange.parse_option_positions()
            tracked_product_ids = self.strategy.state.tracked_product_ids
            if tracked_product_ids:
                legs = [leg for leg in legs if leg.product_id in tracked_product_ids]
            logger.info(f"Parsed {len(legs)} option positions")
        except Exception as e:
            logger.error(f"Failed to parse option positions: {str(e)}")
            raise
        
        total_pnl = 0.0
        positions: list[dict[str, Any]] = []

        for leg in legs:
            try:
                # Add 10 second timeout per quote fetch
                quote = await asyncio.wait_for(
                    exchange.get_best_quote(leg.product_id),
                    timeout=10.0
                )
                qty = abs(leg.size)
                
                # Validate quote data before calculations
                if quote.best_bid is None or quote.best_ask is None:
                    logger.warning(f"Invalid quote for product {leg.product_id}: bid={quote.best_bid}, ask={quote.best_ask}")
                    continue
                
                # Calculate PnL safely
                if leg.size > 0:
                    pnl = qty * leg.contract_value * (float(quote.best_bid) - float(leg.entry_price))
                else:
                    pnl = qty * leg.contract_value * (float(leg.entry_price) - float(quote.best_ask))
                
                total_pnl += pnl
                positions.append(
                    {
                        "product_id": leg.product_id,
                        "symbol": leg.symbol,
                        "option_type": leg.option_type,
                        "strike": leg.strike,
                        "expiry": leg.expiry.isoformat() if leg.expiry is not None else None,
                        "size": leg.size,
                        "entry_price": leg.entry_price,
                        "contract_value": leg.contract_value,
                        "best_bid": quote.best_bid,
                        "best_ask": quote.best_ask,
                        "current_pnl": pnl,
                    }
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching quote for product {leg.product_id}")
                continue
            except Exception as e:
                logger.error(f"Error processing position for product {leg.product_id}: {str(e)}")
                continue

        realized_pnl = self.strategy.state.realized_pnl_total
        combined_pnl = realized_pnl + total_pnl
        self.strategy.state.unrealized_pnl_total = total_pnl
        self.strategy.state.combined_pnl_total = combined_pnl
        logger.info(
            "Summary collected: positions=%s unrealized_pnl=%s realized_pnl=%s combined_pnl=%s",
            len(positions),
            total_pnl,
            realized_pnl,
            combined_pnl,
        )
        
        return {
            "index_price": index_price,
            "unrealized_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "combined_pnl": combined_pnl,
            "strategy_realized_pnl": realized_pnl,
            "closed_positions": self.strategy.positions.closed_positions,
            "closed_position_count": len(self.strategy.positions.closed_positions),
            "profit_exit_threshold_usd": self.strategy.settings.profit_exit_threshold_usd,
            "loss_exit_threshold_usd": self.strategy.settings.loss_exit_threshold_usd,
            "allowed_ratio_spreads": self.strategy.settings.ratio_spreads,
            "position_count": len(positions),
            "positions": positions,
            "strategy_state": self.status()["strategy_state"],
        }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._task is None or self._task.done():
                return {"status": "not_running", "running": False}
            self._task.cancel()
            return {"status": "stopped", "running": False}

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._task is not None and not self._task.done()
        
        # Build call and put ratio states
        call_ratio_state = None
        if self.strategy.state.call_ratio_state is not None:
            call_ratio_state = {
                "status": self.strategy.state.call_ratio_state.status_message,
                "action": self.strategy.state.call_ratio_state.action,
                "trigger_pnl": self.strategy.state.call_ratio_state.trigger_pnl,
            }
        
        put_ratio_state = None
        if self.strategy.state.put_ratio_state is not None:
            put_ratio_state = {
                "status": self.strategy.state.put_ratio_state.status_message,
                "action": self.strategy.state.put_ratio_state.action,
                "trigger_pnl": self.strategy.state.put_ratio_state.trigger_pnl,
            }
        
        return {
            "running": running,
            "strategy_state": {
                "last_index_price": self.strategy.state.last_index_price,
                "status_message": self.strategy.state.status_message,
                "action": self.strategy.state.action,
                "status": self.strategy.state.status,
                "trigger_price": self.strategy.state.trigger_price,
                "trigger_pnl": self.strategy.state.trigger_pnl,
                "realized_pnl": self.strategy.state.realized_pnl_total,
                "realized_pnl_total": self.strategy.state.realized_pnl_total,
                "unrealized_pnl": self.strategy.state.unrealized_pnl_total,
                "combined_pnl": self.strategy.state.combined_pnl_total,
                "profit_exit_threshold_usd": self.strategy.settings.profit_exit_threshold_usd,
                "loss_exit_threshold_usd": self.strategy.settings.loss_exit_threshold_usd,
                "allowed_ratio_spreads": self.strategy.settings.ratio_spreads,
                "call_ratio": call_ratio_state,
                "put_ratio": put_ratio_state,
            },
        }


def build_strategy() -> StrategyEngine:
    settings = load_settings()
    exchange = DeltaExchangeClient(
        api_key=settings.api_key,
        api_secret=settings.api_secret,
        base_url=settings.base_url,
        ssl_verify=settings.ssl_verify,
    )
    positions = PositionManager(exchange)
    executor = OrderExecutor(exchange)
    return StrategyEngine(exchange, positions, executor, settings)


app = FastAPI(title="Delta BTC Options API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

log_handler = JsonLogHandler(max_records=2000)
logging.getLogger().addHandler(log_handler)

strategy = build_strategy()
manager = BotManager(strategy)


@app.on_event("startup")
async def startup_event() -> None:
    setup_logging("INFO")
    root_logger = logging.getLogger()
    if log_handler not in root_logger.handlers:
        root_logger.addHandler(log_handler)
    logging.getLogger(__name__).info("Delta BTC Options API starting")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logging.getLogger(__name__).info("Delta BTC Options API shutting down")
    manager.stop()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/start")
async def api_start() -> dict[str, Any]:
    return manager.start()


@app.post("/stop")
async def api_stop() -> dict[str, Any]:
    return manager.stop()


@app.get("/status")
async def api_status() -> dict[str, Any]:
    return manager.status()


@app.get("/summary")
async def api_summary() -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    try:
        return await manager.collect_summary()
    except Exception as exc:
        logger.exception("Summary API error")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to collect summary: {str(exc)}"
        )


@app.get("/logs")
async def api_logs(limit: int | None = None) -> dict[str, Any]:
    return {"count": len(log_handler.latest(limit)), "logs": log_handler.latest(limit)}
