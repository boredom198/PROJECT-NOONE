"""SQLite Execution Journal for Component 3.

Provides persistent, transactional state storage for order lifecycles, fills,
daily baselines, external transfers, safety freezes, and sync audits.
Strictly conforms to PROJECT NOONE Technical Design Specification V2.2.4.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple


CURRENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class JournalOrder:
    client_order_id: str
    exchange_order_id: Optional[str]
    symbol: str
    action_type: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    reduce_only: bool
    state: str
    reconciliation_state: str
    filled_quantity: float
    cum_quote: float
    avg_price: float
    created_at: str
    updated_at: str
    error_message: Optional[str]


@dataclass(frozen=True)
class JournalTradeFill:
    trade_id: str
    client_order_id: str
    exchange_order_id: Optional[str]
    symbol: str
    price: float
    quantity: float
    commission: float
    commission_asset: str
    realized_pnl: float
    is_maker: bool
    timestamp: int


@dataclass(frozen=True)
class DailyBaselineRecord:
    date_utc: str
    baseline_equity: float
    verified_status: str
    captured_at: str
    note: Optional[str]


@dataclass(frozen=True)
class SafetyFreezeRecord:
    id: int
    reason: str
    active: bool
    details: Optional[str]
    triggered_at: str
    cleared_at: Optional[str]
    clearance_note: Optional[str]


class ExecutionJournal:
    """Manages SQLite storage and atomic transactions for Component 3."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # Autocommit mode; transactions handled explicitly
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an explicit BEGIN IMMEDIATE transaction."""
        conn = self._get_connection()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables, indexes, and schema version record if not present."""
        with self.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            row = conn.execute("SELECT MAX(version) AS ver FROM schema_version;").fetchone()
            current_ver = row["ver"] if row and row["ver"] is not None else 0

            if current_ver < 1:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS orders (
                        client_order_id TEXT PRIMARY KEY,
                        exchange_order_id TEXT,
                        symbol TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        side TEXT NOT NULL,
                        order_type TEXT NOT NULL DEFAULT 'MARKET',
                        quantity REAL NOT NULL,
                        price REAL,
                        reduce_only INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL,
                        reconciliation_state TEXT NOT NULL DEFAULT 'PENDING',
                        filled_quantity REAL NOT NULL DEFAULT 0.0,
                        cum_quote REAL NOT NULL DEFAULT 0.0,
                        avg_price REAL NOT NULL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        error_message TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS order_state_transitions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_order_id TEXT NOT NULL REFERENCES orders(client_order_id),
                        from_state TEXT,
                        to_state TEXT NOT NULL,
                        reason TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trade_fills (
                        trade_id TEXT PRIMARY KEY,
                        client_order_id TEXT NOT NULL REFERENCES orders(client_order_id),
                        exchange_order_id TEXT,
                        symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        commission REAL NOT NULL DEFAULT 0.0,
                        commission_asset TEXT NOT NULL DEFAULT 'USDT',
                        realized_pnl REAL NOT NULL DEFAULT 0.0,
                        is_maker INTEGER NOT NULL DEFAULT 0,
                        timestamp INTEGER NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_baselines (
                        date_utc TEXT PRIMARY KEY,
                        baseline_equity REAL NOT NULL,
                        verified_status TEXT NOT NULL,
                        captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        note TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS external_transfers (
                        tran_id TEXT PRIMARY KEY,
                        income_type TEXT NOT NULL,
                        asset TEXT NOT NULL,
                        income REAL NOT NULL,
                        timestamp INTEGER NOT NULL,
                        is_external INTEGER NOT NULL DEFAULT 1,
                        note TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS safety_freezes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        reason TEXT NOT NULL,
                        active INTEGER NOT NULL DEFAULT 1,
                        details TEXT,
                        triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        cleared_at TIMESTAMP,
                        clearance_note TEXT
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operator_acknowledgements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        acknowledgement_type TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        operator_name TEXT NOT NULL,
                        details TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sync_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sync_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        details TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS order_recovery_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_order_id TEXT NOT NULL REFERENCES orders(client_order_id),
                        step INTEGER NOT NULL,
                        step_name TEXT NOT NULL,
                        result TEXT NOT NULL,
                        details TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )

                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state);")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_orders_reconciliation ON orders(reconciliation_state);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_order_transitions_coid ON order_state_transitions(client_order_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trade_fills_coid ON trade_fills(client_order_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_safety_freezes_active ON safety_freezes(active);"
                )

                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, CURRENT_TIMESTAMP);",
                    (1,),
                )

    # -------------------------------------------------------------------------
    # Order CRUD & State Transitions
    # -------------------------------------------------------------------------

    def create_order_intent(
        self,
        client_order_id: str,
        symbol: str,
        action_type: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> JournalOrder:
        """Atomically persist order intent in CREATED state before socket transmission."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    client_order_id, symbol, action_type, side, order_type,
                    quantity, price, reduce_only, state, reconciliation_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'CREATED', 'PENDING');
                """,
                (
                    client_order_id,
                    symbol,
                    action_type,
                    side,
                    order_type,
                    quantity,
                    price,
                    1 if reduce_only else 0,
                ),
            )
            conn.execute(
                """
                INSERT INTO order_state_transitions (client_order_id, from_state, to_state, reason)
                VALUES (?, NULL, 'CREATED', 'Intent persisted before socket write');
                """,
                (client_order_id,),
            )
        return self.get_order(client_order_id)  # type: ignore

    def transition_order_state(
        self,
        client_order_id: str,
        to_state: str,
        reason: Optional[str] = None,
        exchange_order_id: Optional[str] = None,
        filled_quantity: Optional[float] = None,
        cum_quote: Optional[float] = None,
        avg_price: Optional[float] = None,
        error_message: Optional[str] = None,
        reconciliation_state: Optional[str] = None,
    ) -> None:
        """Record an atomic state transition for an order."""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM orders WHERE client_order_id = ?;", (client_order_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Order {client_order_id} does not exist.")
            from_state = row["state"]

            updates = ["state = ?", "updated_at = CURRENT_TIMESTAMP"]
            params: List[Any] = [to_state]

            if exchange_order_id is not None:
                updates.append("exchange_order_id = ?")
                params.append(str(exchange_order_id))
            if filled_quantity is not None:
                updates.append("filled_quantity = ?")
                params.append(filled_quantity)
            if cum_quote is not None:
                updates.append("cum_quote = ?")
                params.append(cum_quote)
            if avg_price is not None:
                updates.append("avg_price = ?")
                params.append(avg_price)
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)
            if reconciliation_state is not None:
                updates.append("reconciliation_state = ?")
                params.append(reconciliation_state)

            params.append(client_order_id)
            conn.execute(
                f"UPDATE orders SET {', '.join(updates)} WHERE client_order_id = ?;",
                params,
            )
            conn.execute(
                """
                INSERT INTO order_state_transitions (client_order_id, from_state, to_state, reason)
                VALUES (?, ?, ?, ?);
                """,
                (client_order_id, from_state, to_state, reason),
            )

    def get_order(self, client_order_id: str) -> Optional[JournalOrder]:
        """Fetch order record by client_order_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?;", (client_order_id,)
            ).fetchone()
            if not row:
                return None
            return JournalOrder(
                client_order_id=row["client_order_id"],
                exchange_order_id=row["exchange_order_id"],
                symbol=row["symbol"],
                action_type=row["action_type"],
                side=row["side"],
                order_type=row["order_type"],
                quantity=row["quantity"],
                price=row["price"],
                reduce_only=bool(row["reduce_only"]),
                state=row["state"],
                reconciliation_state=row["reconciliation_state"],
                filled_quantity=row["filled_quantity"],
                cum_quote=row["cum_quote"],
                avg_price=row["avg_price"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                error_message=row["error_message"],
            )

    def get_unresolved_orders(self) -> List[JournalOrder]:
        """Return all orders that are non-terminal or reconciliation PENDING/DISCREPANCY."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM orders 
                WHERE state IN ('CREATED', 'DISPATCHING', 'ACKNOWLEDGED', 'OPEN', 'PARTIALLY_FILLED', 'UNKNOWN')
                   OR reconciliation_state IN ('PENDING', 'DISCREPANCY')
                ORDER BY created_at ASC;
                """
            ).fetchall()
            return [
                JournalOrder(
                    client_order_id=r["client_order_id"],
                    exchange_order_id=r["exchange_order_id"],
                    symbol=r["symbol"],
                    action_type=r["action_type"],
                    side=r["side"],
                    order_type=r["order_type"],
                    quantity=r["quantity"],
                    price=r["price"],
                    reduce_only=bool(r["reduce_only"]),
                    state=r["state"],
                    reconciliation_state=r["reconciliation_state"],
                    filled_quantity=r["filled_quantity"],
                    cum_quote=r["cum_quote"],
                    avg_price=r["avg_price"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    error_message=r["error_message"],
                )
                for r in rows
            ]

    # -------------------------------------------------------------------------
    # Trade Fills Deduplication
    # -------------------------------------------------------------------------

    def record_trade_fill(
        self,
        trade_id: str,
        client_order_id: str,
        symbol: str,
        price: float,
        quantity: float,
        commission: float,
        commission_asset: str,
        realized_pnl: float,
        timestamp: int,
        exchange_order_id: Optional[str] = None,
        is_maker: bool = False,
    ) -> bool:
        """Insert trade fill deduplicated by trade_id. Returns True if inserted, False if duplicate."""
        with self.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO trade_fills (
                        trade_id, client_order_id, exchange_order_id, symbol,
                        price, quantity, commission, commission_asset,
                        realized_pnl, is_maker, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(trade_id),
                        client_order_id,
                        str(exchange_order_id) if exchange_order_id is not None else None,
                        symbol,
                        price,
                        quantity,
                        commission,
                        commission_asset,
                        realized_pnl,
                        1 if is_maker else 0,
                        int(timestamp),
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                # Already recorded
                return False

    def get_order_fills(self, client_order_id: str) -> List[JournalTradeFill]:
        """Fetch all fills associated with a client_order_id."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_fills WHERE client_order_id = ? ORDER BY timestamp ASC;",
                (client_order_id,),
            ).fetchall()
            return [
                JournalTradeFill(
                    trade_id=r["trade_id"],
                    client_order_id=r["client_order_id"],
                    exchange_order_id=r["exchange_order_id"],
                    symbol=r["symbol"],
                    price=r["price"],
                    quantity=r["quantity"],
                    commission=r["commission"],
                    commission_asset=r["commission_asset"],
                    realized_pnl=r["realized_pnl"],
                    is_maker=bool(r["is_maker"]),
                    timestamp=r["timestamp"],
                )
                for r in rows
            ]

    # -------------------------------------------------------------------------
    # Daily Baselines
    # -------------------------------------------------------------------------

    def record_daily_baseline(
        self,
        date_utc: str,
        baseline_equity: float,
        verified_status: str,
        note: Optional[str] = None,
    ) -> None:
        """Persist or update UTC midnight baseline record."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO daily_baselines (date_utc, baseline_equity, verified_status, captured_at, note)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(date_utc) DO UPDATE SET
                    baseline_equity = excluded.baseline_equity,
                    verified_status = excluded.verified_status,
                    captured_at = CURRENT_TIMESTAMP,
                    note = excluded.note;
                """,
                (date_utc, baseline_equity, verified_status, note),
            )

    def get_daily_baseline(self, date_utc: str) -> Optional[DailyBaselineRecord]:
        """Fetch daily baseline for a given UTC date (YYYY-MM-DD)."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_baselines WHERE date_utc = ?;", (date_utc,)
            ).fetchone()
            if not row:
                return None
            return DailyBaselineRecord(
                date_utc=row["date_utc"],
                baseline_equity=row["baseline_equity"],
                verified_status=row["verified_status"],
                captured_at=row["captured_at"],
                note=row["note"],
            )

    # -------------------------------------------------------------------------
    # External Transfers
    # -------------------------------------------------------------------------

    def record_external_transfer(
        self,
        tran_id: str,
        income_type: str,
        asset: str,
        income: float,
        timestamp: int,
        is_external: bool = True,
        note: Optional[str] = None,
    ) -> bool:
        """Insert capital transfer deduplicated on tran_id. Returns True if inserted, False if duplicate."""
        with self.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO external_transfers (
                        tran_id, income_type, asset, income, timestamp, is_external, note
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(tran_id),
                        income_type,
                        asset,
                        income,
                        int(timestamp),
                        1 if is_external else 0,
                        note,
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_net_external_transfers(self, since_timestamp_ms: int) -> float:
        """Sum net external capital transfers in USDT since given timestamp."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(income), 0.0) AS net_transfers
                FROM external_transfers
                WHERE asset = 'USDT'
                  AND income_type = 'TRANSFER'
                  AND is_external = 1
                  AND timestamp >= ?;
                """,
                (since_timestamp_ms,),
            ).fetchone()
            return float(row["net_transfers"]) if row else 0.0

    # -------------------------------------------------------------------------
    # Safety Freezes
    # -------------------------------------------------------------------------

    def trigger_safety_freeze(self, reason: str, details: Optional[str] = None) -> int:
        """Trigger an active safety freeze if not already active for this reason."""
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT id FROM safety_freezes WHERE reason = ? AND active = 1;", (reason,)
            ).fetchone()
            if existing:
                return int(existing["id"])
            cursor = conn.execute(
                """
                INSERT INTO safety_freezes (reason, active, details, triggered_at)
                VALUES (?, 1, ?, CURRENT_TIMESTAMP);
                """,
                (reason, details),
            )
            return int(cursor.lastrowid)

    def clear_safety_freeze(self, reason: str, clearance_note: Optional[str] = None) -> None:
        """Clear active safety freeze for a given reason."""
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE safety_freezes
                SET active = 0, cleared_at = CURRENT_TIMESTAMP, clearance_note = ?
                WHERE reason = ? AND active = 1;
                """,
                (clearance_note, reason),
            )

    def get_active_safety_freezes(self) -> List[SafetyFreezeRecord]:
        """Fetch all currently active safety freezes."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM safety_freezes WHERE active = 1 ORDER BY triggered_at ASC;"
            ).fetchall()
            return [
                SafetyFreezeRecord(
                    id=r["id"],
                    reason=r["reason"],
                    active=bool(r["active"]),
                    details=r["details"],
                    triggered_at=r["triggered_at"],
                    cleared_at=r["cleared_at"],
                    clearance_note=r["clearance_note"],
                )
                for r in rows
            ]

    def has_active_freeze(self, reason: Optional[str] = None) -> bool:
        """Check whether any freeze (or a specific reason) is active."""
        with self._get_connection() as conn:
            if reason:
                row = conn.execute(
                    "SELECT 1 FROM safety_freezes WHERE active = 1 AND reason = ? LIMIT 1;",
                    (reason,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM safety_freezes WHERE active = 1 LIMIT 1;"
                ).fetchone()
            return row is not None

    # -------------------------------------------------------------------------
    # Audit & Recovery Logging
    # -------------------------------------------------------------------------

    def log_sync_audit(self, sync_type: str, status: str, details: Optional[str] = None) -> None:
        """Record an account or position sync audit event."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sync_audit_log (sync_type, status, details, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (sync_type, status, details),
            )

    def log_recovery_attempt(
        self,
        client_order_id: str,
        step: int,
        step_name: str,
        result: str,
        details: Optional[str] = None,
    ) -> None:
        """Record an order recovery step outcome."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO order_recovery_attempts (
                    client_order_id, step, step_name, result, details, timestamp
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (client_order_id, step, step_name, result, details),
            )

    def record_operator_acknowledgement(
        self,
        acknowledgement_type: str,
        reason: str,
        operator_name: str,
        details: Optional[str] = None,
    ) -> None:
        """Record an operator acknowledgement entry."""
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO operator_acknowledgements (
                    acknowledgement_type, reason, operator_name, details, timestamp
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (acknowledgement_type, reason, operator_name, details),
            )

