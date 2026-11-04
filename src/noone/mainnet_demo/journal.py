"""Persistent SQLite Journal and State Lock for Mainnet Demo Execution Mode."""
from contextlib import contextmanager
from datetime import datetime, timezone
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

from noone.mainnet_demo.config import DemoState, MainnetDemoError, MainnetDemoLockedError


def _finite_number(value: Any, name: str, *, minimum: Optional[float] = None,
                   strictly_positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    result = float(value)
    if strictly_positive and result <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return result


class MainnetDemoJournal:
    """Manages SQLite persistence for Mainnet demo states, orders, and loss budgets."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = FULL;")
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS demo_state_lock (
                id INTEGER PRIMARY KEY CHECK (id = 1), state TEXT NOT NULL,
                is_locked INTEGER NOT NULL DEFAULT 0, lock_reason TEXT, updated_at TEXT NOT NULL);""")
            conn.execute("""CREATE TABLE IF NOT EXISTS demo_loss_budget (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cumulative_realized_loss_usdt REAL NOT NULL DEFAULT 0.0, updated_at TEXT NOT NULL);""")
            conn.execute("""CREATE TABLE IF NOT EXISTS demo_orders (
                client_order_id TEXT PRIMARY KEY, exchange_order_id TEXT, order_type TEXT NOT NULL,
                side TEXT NOT NULL, requested_qty REAL NOT NULL, filled_qty REAL NOT NULL DEFAULT 0.0,
                price REAL, avg_price REAL, fee REAL NOT NULL DEFAULT 0.0, status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);""")
            conn.execute("""CREATE TABLE IF NOT EXISTS demo_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, side TEXT NOT NULL,
                entry_price REAL NOT NULL, exit_price REAL, quantity REAL NOT NULL,
                realized_pnl REAL, fees REAL NOT NULL DEFAULT 0.0, status TEXT NOT NULL,
                created_at TEXT NOT NULL, closed_at TEXT);""")
            conn.execute("""CREATE TABLE IF NOT EXISTS demo_execution_claim (
                id INTEGER PRIMARY KEY CHECK (id = 1), runner_id TEXT NOT NULL,
                claimed_at TEXT NOT NULL, status TEXT NOT NULL);""")
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("INSERT OR IGNORE INTO demo_state_lock (id,state,is_locked,lock_reason,updated_at) VALUES (1,'DISARMED',0,NULL,?);", (now,))
            conn.execute("INSERT OR IGNORE INTO demo_loss_budget (id,cumulative_realized_loss_usdt,updated_at) VALUES (1,0.0,?);", (now,))

    # State and lock management
    def get_state(self) -> DemoState:
        with self.connection() as conn:
            row = conn.execute("SELECT state FROM demo_state_lock WHERE id=1").fetchone()
            return DemoState(row["state"]) if row else DemoState.DISARMED

    def is_locked(self) -> bool:
        with self.connection() as conn:
            row = conn.execute("SELECT is_locked FROM demo_state_lock WHERE id=1").fetchone()
            return bool(row["is_locked"]) if row else False

    def get_lock_reason(self) -> Optional[str]:
        with self.connection() as conn:
            row = conn.execute("SELECT lock_reason FROM demo_state_lock WHERE id=1").fetchone()
            return row["lock_reason"] if row else None

    def set_state(self, state: DemoState, lock_reason: Optional[str] = None) -> None:
        """Atomically update state; a permanent LOCKED state cannot be cleared."""
        if not isinstance(state, DemoState):
            raise ValueError(f"state must be DemoState, got {state!r}")
        if lock_reason is not None and not isinstance(lock_reason, str):
            raise ValueError("lock_reason must be a string or None")
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = conn.execute("SELECT is_locked FROM demo_state_lock WHERE id=1").fetchone()
                if current is None:
                    raise MainnetDemoError("Demo state row is missing")
                if current["is_locked"] and state != DemoState.LOCKED:
                    conn.execute("COMMIT")
                    return
                conn.execute("""UPDATE demo_state_lock SET state=?,
                    is_locked=CASE WHEN is_locked=1 THEN 1 ELSE ? END,
                    lock_reason=CASE WHEN lock_reason IS NOT NULL AND ? IS NULL THEN lock_reason ELSE ? END,
                    updated_at=? WHERE id=1""",
                    (state.value, 1 if state == DemoState.LOCKED else 0, lock_reason, lock_reason, now))
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            if state == DemoState.LOCKED:
                # FULL synchronous commit is the durability guarantee; checkpoint is housekeeping.
                try:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass

    def trigger_permanent_lock(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A non-empty lock reason is required")
        self.set_state(DemoState.LOCKED, lock_reason=reason.strip())

    # Atomic single-execution claim
    def claim_execution(self, runner_id: str) -> bool:
        if not isinstance(runner_id, str) or not runner_id.strip():
            raise ValueError("runner_id must be a non-empty string")
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                lock = conn.execute("SELECT is_locked,lock_reason FROM demo_state_lock WHERE id=1").fetchone()
                if lock is None:
                    raise MainnetDemoError("Demo state row is missing")
                if lock["is_locked"]:
                    raise MainnetDemoLockedError(f"Execution claim rejected: Demo is permanently LOCKED ({lock['lock_reason']})")
                claim = conn.execute("SELECT runner_id,status FROM demo_execution_claim WHERE id=1").fetchone()
                if claim:
                    raise MainnetDemoLockedError(
                        f"Execution claim rejected: Already claimed by {claim['runner_id']} with status '{claim['status']}'.")
                conn.execute("INSERT INTO demo_execution_claim (id,runner_id,claimed_at,status) VALUES (1,?,?, 'CLAIMED')", (runner_id.strip(), now))
                conn.execute("COMMIT")
                return True
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    def get_execution_claim(self) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM demo_execution_claim WHERE id=1").fetchone()
            return dict(row) if row else None

    def update_claim_status(self, runner_id: str, status: str) -> None:
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a non-empty string")
        with self.connection() as conn:
            cur = conn.execute("UPDATE demo_execution_claim SET status=? WHERE id=1 AND runner_id=?", (status.strip().upper(), runner_id))
            if cur.rowcount != 1:
                raise MainnetDemoError(f"Execution claim for runner_id {runner_id!r} was not found")

    # Loss budget: this method adds losses only; gains are not netted against the safety budget.
    def get_cumulative_loss(self) -> float:
        with self.connection() as conn:
            row = conn.execute("SELECT cumulative_realized_loss_usdt FROM demo_loss_budget WHERE id=1").fetchone()
            value = float(row[0]) if row else 0.0
            if not math.isfinite(value) or value < 0:
                raise MainnetDemoError("Stored cumulative realized loss is invalid")
            return value

    def record_realized_loss(self, loss_usdt: float) -> float:
        loss = _finite_number(loss_usdt, "loss_usdt", minimum=0.0)
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute("UPDATE demo_loss_budget SET cumulative_realized_loss_usdt=cumulative_realized_loss_usdt+?, updated_at=? WHERE id=1", (loss, now))
                if cur.rowcount != 1:
                    raise MainnetDemoError("Loss-budget row is missing")
                total = float(conn.execute("SELECT cumulative_realized_loss_usdt FROM demo_loss_budget WHERE id=1").fetchone()[0])
                if not math.isfinite(total):
                    raise MainnetDemoError("Cumulative realized loss became non-finite")
                conn.execute("COMMIT")
                return total
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    def get_remaining_budget(self, max_budget_usdt: float) -> float:
        maximum = _finite_number(max_budget_usdt, "max_budget_usdt", strictly_positive=True)
        return max(0.0, maximum - self.get_cumulative_loss())

    # Order tracking
    def record_order(self, client_order_id: str, order_type: str, side: str,
                     requested_qty: float, price: Optional[float] = None,
                     status: str = "SUBMITTED", exchange_order_id: Optional[str] = None) -> None:
        if not isinstance(client_order_id, str) or not client_order_id.strip():
            raise ValueError("client_order_id must be a non-empty string")
        if not isinstance(order_type, str) or not order_type.strip():
            raise ValueError("order_type must be a non-empty string")
        if not isinstance(side, str) or side.upper() not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        qty = _finite_number(requested_qty, "requested_qty", strictly_positive=True)
        px = None if price is None else _finite_number(price, "price", strictly_positive=True)
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a non-empty string")
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self.connection() as conn:
                conn.execute("""INSERT INTO demo_orders
                    (client_order_id,exchange_order_id,order_type,side,requested_qty,filled_qty,price,avg_price,fee,status,created_at,updated_at)
                    VALUES (?,?,?,?,?,0.0,?,NULL,0.0,?,?,?)""",
                    (client_order_id.strip(), exchange_order_id, order_type.strip().upper(), side.upper(), qty, px, status.strip().upper(), now, now))
        except sqlite3.IntegrityError as exc:
            raise MainnetDemoError(f"Duplicate or invalid journal order {client_order_id!r}: {exc}") from exc

    def update_order(self, client_order_id: str, status: str,
                     filled_qty: Optional[float] = None, avg_price: Optional[float] = None,
                     fee: Optional[float] = None, exchange_order_id: Optional[str] = None) -> None:
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a non-empty string")
        filled = None if filled_qty is None else _finite_number(filled_qty, "filled_qty", minimum=0.0)
        avg = None if avg_price is None else _finite_number(avg_price, "avg_price", strictly_positive=True)
        fees = None if fee is None else _finite_number(fee, "fee", minimum=0.0)
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cur = conn.execute("""UPDATE demo_orders SET status=?, filled_qty=COALESCE(?,filled_qty),
                avg_price=COALESCE(?,avg_price), fee=COALESCE(?,fee),
                exchange_order_id=COALESCE(?,exchange_order_id), updated_at=? WHERE client_order_id=?""",
                (status.strip().upper(), filled, avg, fees, exchange_order_id, now, client_order_id))
            if cur.rowcount != 1:
                raise MainnetDemoError(f"Cannot update missing journal order {client_order_id!r}")

    def get_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM demo_orders WHERE client_order_id=?", (client_order_id,)).fetchone()
            return dict(row) if row else None

    def get_unresolved_orders(self) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM demo_orders WHERE status IN ('SUBMITTED','OPEN','PARTIALLY_FILLED','UNKNOWN')").fetchall()
            return [dict(row) for row in rows]

    # Position tracking
    def record_open_position(self, symbol: str, side: str, entry_price: float,
                             quantity: float, fee: float) -> int:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(side, str) or side.upper() not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        entry = _finite_number(entry_price, "entry_price", strictly_positive=True)
        qty = _finite_number(quantity, "quantity", strictly_positive=True)
        fees = _finite_number(fee, "fee", minimum=0.0)
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cur = conn.execute("INSERT INTO demo_positions (symbol,side,entry_price,quantity,fees,status,created_at) VALUES (?,?,?,?,?,'OPEN',?)",
                               (symbol.strip(), side.upper(), entry, qty, fees, now))
            return int(cur.lastrowid)

    def record_closed_position(self, pos_id: int, exit_price: float,
                               realized_pnl: float, exit_fee: float) -> None:
        if not isinstance(pos_id, int) or isinstance(pos_id, bool) or pos_id <= 0:
            raise ValueError("pos_id must be a positive integer")
        exit_px = _finite_number(exit_price, "exit_price", strictly_positive=True)
        pnl = _finite_number(realized_pnl, "realized_pnl")
        fees = _finite_number(exit_fee, "exit_fee", minimum=0.0)
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            cur = conn.execute("""UPDATE demo_positions SET exit_price=?, realized_pnl=?, fees=fees+?,
                status='CLOSED', closed_at=? WHERE id=? AND status='OPEN'""",
                (exit_px, pnl, fees, now, pos_id))
            if cur.rowcount != 1:
                raise MainnetDemoError(f"Position {pos_id} does not exist or is not OPEN; refusing duplicate/invalid close")
