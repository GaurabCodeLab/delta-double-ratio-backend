# Delta BTC Options Ratio-Spread Bot

Python async trading bot for Delta Exchange BTC options that:

- Detects an existing configured ratio spread automatically
- Continuously monitors overall ratio spread P&L
- Closes all open ratio spread positions immediately when PnL reaches the configured profit or loss thresholds

## Key Features

- Modular architecture:
  - `exchange_client.py` (Delta API wrapper + retries)
  - `position_manager.py` (position discovery and live state)
  - `strategy_engine.py` (trigger and exit logic)
  - `order_executor.py` (market order placement and fill confirmation)
- Uses Delta index price (`BTCUSDT`) as the trigger reference
- Handles partial fills via fill reconciliation loops
- Retries transient API failures with exponential backoff
- Detailed structured logging

## Strategy Behavior

### 1) Initial Position Discovery

The bot scans open option positions and identifies a valid ratio spread where all legs have:

- same expiry
- same option type (`call` or `put`)
- net structure: long and short legs matching a configured ratio such as `1:2`, `1:3`, `1:5`, or any user-defined ratio

### 2) Monitoring and Exit

Once a configured ratio spread is found, the bot:

- continuously monitors the overall unrealized PnL of the active ratio spread
- reconciles observed position reductions against Delta fill history and includes gross realized PnL in its exit total
- does not rebalance, roll, convert, or open new offsetting positions
- closes all open ratio spread positions immediately when either condition is met:
  - profit reaches the configured profit threshold
  - loss reaches the configured loss threshold

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy env template and set credentials:

```bash
copy .env.example .env
```

3. Run bot:

```bash
python -m src.main
```

## Environment Variables

- `DELTA_API_KEY`
- `DELTA_API_SECRET`
- `DELTA_BASE_URL` (default: `https://cdn-ind.testnet.deltaex.org`)
- `DELTA_SSL_VERIFY` (default: `false` for testnet environments with missing CA chain)
- `POLL_INTERVAL_SECONDS` (default: `2`)
- `PNL_PROFIT_THRESHOLD_USD` (default: `10`)
- `PNL_LOSS_THRESHOLD_USD` (default: `-20`)
- `RATIO_SPREADS` (default: `1:2`). Supports multiple comma-separated values like `1:2,1:3,1:5`.

Heartbeat logs run on the same interval as `POLL_INTERVAL_SECONDS`.

Realized PnL tracking is in-memory for the lifetime of the bot process. It requires identifiable, timestamped fills from Delta's fills API; if fill history cannot cover an observed reduction, the bot does not use an incomplete PnL total to trigger an exit. Commissions and fees are not included in the realized PnL calculation.

The `/summary` response includes `closed_positions` and `closed_position_count`. Each closed-position record is fill-level and includes `symbol`, `type`, `strike`, `expiry`, `side`, signed `size`, `entry`, `closed_price`, `closed_time`, and `realized_pnl`, along with fill/product IDs and additional descriptive aliases. Records are retained in memory for the bot process lifetime.

## Testing

After installing dependencies, you can run the test suite with:

python -m pytest

To generate a coverage report for the source code, run:

```bash
python -m pytest --cov=src --cov-report=term-missing
```

You can also run only specific test groups using markers:

```bash
python -m pytest -m unit -q
python -m pytest -m api -q
python -m pytest -m integration -q
python -m pytest -m async -q
```

## Notes

- The code uses `delta-rest-client` and wraps sync calls with `asyncio.to_thread`.
- Delta REST field names can vary by account/product type. Mapping logic is centralized in `src/models.py` and `src/exchange_client.py`.
- Test only on testnet first.
