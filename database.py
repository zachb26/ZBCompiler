# -*- coding: utf-8 -*-
"""
database.py — DatabaseManager class and get_database_manager() factory.

Handles SQLite and PostgreSQL backends. All DB I/O for the analysis table,
portfolio memberships, and decision log lives here.
"""
import copy
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import shutil
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import psycopg
import streamlit as st

from constants import *
from utils_fmt import *
from utils_time import *
from exports import *
from cache import *


class DatabaseManager:
    def __init__(self, db_name):
        self._raw_db_name = str(db_name).strip()
        self.db_path = None
        self._write_lock = threading.RLock()
        self._backend = "sqlite"
        self._sqlite_target = None
        self._sqlite_uri = False
        self._anchor_connection = None
        self._fallback_notice = None
        self._storage_mode = "disk"
        self._postgres_dsn = None
        self._postgres_optional_table_errors = {}
        self._initialize_storage_target()
        self.create_tables()

    def _initialize_storage_target(self):
        if is_postgres_database_url(self._raw_db_name):
            self._backend = "postgres"
            self._postgres_dsn = self._raw_db_name
            self.db_path = None
            self._storage_mode = "server"
            return

        if self._raw_db_name.lower() == ":memory:":
            self._activate_in_memory_mode(
                "Research storage is running in memory only. Library changes will reset when the app stops."
            )
            return

        self.db_path = Path(self._raw_db_name)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_target = str(self.db_path)
        self._sqlite_uri = False
        self._storage_mode = "disk"

    def _activate_in_memory_mode(self, notice):
        self.db_path = None
        self._backend = "sqlite"
        self._sqlite_target = "file:zb_compiler_shared_memory?mode=memory&cache=shared"
        self._sqlite_uri = True
        self._storage_mode = "memory"
        self._fallback_notice = notice
        if self._anchor_connection is None:
            self._anchor_connection = sqlite3.connect(
                self._sqlite_target,
                timeout=30,
                check_same_thread=False,
                uri=True,
            )
            self._configure_connection(self._anchor_connection)

    def _configure_connection(self, conn):
        if self._backend != "sqlite":
            return
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            conn.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.DatabaseError:
            pass

        journal_modes = ["MEMORY"] if self._storage_mode == "memory" else ["WAL", "TRUNCATE", "DELETE"]
        for mode in journal_modes:
            try:
                conn.execute(f"PRAGMA journal_mode={mode}")
                return
            except sqlite3.DatabaseError:
                continue

    def _ph(self):
        return "%s" if self._backend == "postgres" else "?"

    def _connect(self, allow_recover=True):
        if self._backend == "postgres":
            try:
                conn = psycopg.connect(self._postgres_dsn)
            except psycopg.OperationalError as exc:
                raise psycopg.OperationalError(
                    build_postgres_connection_error_message(self._postgres_dsn, exc)
                ) from None
            conn.autocommit = False
            return conn

        # Open a fresh connection per operation so each session sees other users' commits.
        conn = None
        try:
            conn = sqlite3.connect(
                self._sqlite_target,
                timeout=30,
                check_same_thread=False,
                uri=self._sqlite_uri,
            )
            self._configure_connection(conn)
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            return conn
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                conn.close()
            if allow_recover and self._storage_mode == "disk" and self._recover_database_file(exc):
                return self._connect(allow_recover=False)
            if allow_recover and self._storage_mode == "disk" and self._enable_in_memory_fallback(exc):
                return self._connect(allow_recover=False)
            raise

    def _enable_in_memory_fallback(self, exc):
        if self._backend != "sqlite" or self._storage_mode == "memory":
            return False
        message = summarize_fetch_error(exc)
        self._activate_in_memory_mode(
            "Persistent research storage was unavailable, so the app fell back to in-memory mode. "
            f"SQLite reported: {message}"
        )
        return True

    def _recover_database_file(self, exc):
        if self._backend != "sqlite":
            return False
        if self.db_path is None or not self.db_path.exists():
            return False

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if self.db_path.stat().st_size == 0:
                self.db_path.unlink()
            else:
                backup_path = self.db_path.with_name(f"{self.db_path.stem}.corrupt-{timestamp}{self.db_path.suffix}")
                shutil.move(str(self.db_path), str(backup_path))
            return True
        except OSError:
            return False

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _append_storage_notice(self, message):
        text = str(message or "").strip()
        if not text:
            return
        if not self._fallback_notice:
            self._fallback_notice = text
        elif text not in self._fallback_notice:
            self._fallback_notice = f"{self._fallback_notice} {text}"

    def _quote_postgres_identifier(self, identifier):
        return '"' + str(identifier).replace('"', '""') + '"'

    def _get_postgres_table_columns(self, conn, table_name):
        rows = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (str(table_name).strip(),),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _postgres_column_has_unique_constraint(self, conn, table_name, column_name):
        row = conn.execute(
            """
            SELECT 1
            FROM pg_constraint constraint_def
            JOIN pg_class table_def ON table_def.oid = constraint_def.conrelid
            JOIN pg_namespace schema_def ON schema_def.oid = table_def.relnamespace
            JOIN pg_attribute attr
                ON attr.attrelid = table_def.oid
               AND attr.attnum = ANY(constraint_def.conkey)
            WHERE schema_def.nspname = 'public'
              AND table_def.relname = %s
              AND attr.attname = %s
              AND constraint_def.contype IN ('p', 'u')
            LIMIT 1
            """,
            (str(table_name).strip(), str(column_name).strip()),
        ).fetchone()
        return row is not None

    def _normalize_postgres_analysis_columns(self, conn):
        existing_columns = self._get_postgres_table_columns(conn, "analysis")
        existing_lookup = {column_name.lower(): column_name for column_name in existing_columns}
        for column_name in ANALYSIS_COLUMNS:
            existing_name = existing_lookup.get(column_name.lower())
            if not existing_name or existing_name == column_name or column_name in existing_columns:
                continue
            conn.execute(
                f"ALTER TABLE analysis RENAME COLUMN "
                f"{self._quote_postgres_identifier(existing_name)} TO {self._quote_postgres_identifier(column_name)}"
            )
            existing_columns = [column_name if value == existing_name else value for value in existing_columns]
            existing_lookup[column_name.lower()] = column_name
        return set(existing_columns)

    def _ensure_postgres_analysis_schema(self, conn):
        column_sql = ",\n                ".join(
            f'"{name}" {self._postgres_column_definition(definition)}'
            for name, definition in ANALYSIS_COLUMNS.items()
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS analysis (
                {column_sql}
            )
            """
        )

        existing_columns = self._normalize_postgres_analysis_columns(conn)
        existing_lookup = {column_name.lower() for column_name in existing_columns}
        for column_name, definition in ANALYSIS_COLUMNS.items():
            if column_name.lower() in existing_lookup:
                continue
            quoted_name = self._quote_postgres_identifier(column_name)
            if definition == "TEXT PRIMARY KEY":
                # Legacy Postgres installs may already have a different primary key.
                # Add the canonical ticker column without forcing a second PK.
                conn.execute(f"ALTER TABLE analysis ADD COLUMN {quoted_name} TEXT")
            else:
                conn.execute(
                    f"ALTER TABLE analysis ADD COLUMN {quoted_name} {self._postgres_column_definition(definition)}"
                )
            existing_columns.add(column_name)
            existing_lookup.add(column_name.lower())

        if "Ticker" in existing_columns and not self._postgres_column_has_unique_constraint(conn, "analysis", "Ticker"):
            conn.execute(
                'CREATE UNIQUE INDEX IF NOT EXISTS analysis_ticker_unique_idx ON analysis ("Ticker")'
            )

    def _mark_postgres_optional_table_unavailable(self, table_name, exc):
        message = summarize_fetch_error(exc)
        if self._postgres_optional_table_errors.get(table_name) == message:
            return
        self._postgres_optional_table_errors[table_name] = message
        feature_name = str(table_name).replace("_", " ")
        self._append_storage_notice(
            f"Postgres {feature_name} features are unavailable right now. Error: {message}"
        )

    def _ensure_postgres_optional_table(self, table_name, ddl):
        try:
            with self._connection() as conn:
                conn.execute(ddl)
            self._postgres_optional_table_errors.pop(table_name, None)
        except psycopg.Error as exc:
            self._mark_postgres_optional_table_unavailable(table_name, exc)

    _SECTOR_MIGRATION = {
        "Technology":             "Information Technology",
        "Communication Services": "Information Technology",
        "Financial Services":     "Financials",
        "Energy":                 "IMEU",
        "Industrials":            "IMEU",
        "Utilities":              "IMEU",
        "Basic Materials":        "IMEU",
        "Consumer Cyclical":      "Consumer Goods",
        "Consumer Defensive":     "Consumer Goods",
    }

    def _migrate_sector_names(self, conn):
        """Remap legacy yfinance sector names to consolidated OSIG sector names."""
        old_names = list(self._SECTOR_MIGRATION.keys())
        placeholders = ",".join("?" if self._backend == "sqlite" else "%s" for _ in old_names)
        col = '"Sector"' if self._backend == "postgres" else "Sector"
        case_clauses = "\n            ".join(
            f"WHEN {repr(old)} THEN {repr(new)}" for old, new in self._SECTOR_MIGRATION.items()
        )
        sql = f"""
            UPDATE analysis SET {col} = CASE {col}
            {case_clauses}
            ELSE {col}
            END
            WHERE {col} IN ({placeholders})
        """
        conn.execute(sql, old_names)

    def create_tables(self):
        with self._write_lock:
            if self._backend == "postgres":
                with self._connection() as conn:
                    self._ensure_postgres_analysis_schema(conn)
                    self._migrate_sector_names(conn)
                self._ensure_postgres_optional_table(
                    "portfolio_memberships",
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_memberships (
                        ticker TEXT NOT NULL,
                        portfolio TEXT NOT NULL,
                        PRIMARY KEY (ticker, portfolio)
                    )
                    """,
                )
                self._ensure_postgres_optional_table(
                    "decision_log",
                    """
                    CREATE TABLE IF NOT EXISTS decision_log (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        portfolio TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        recommendation TEXT NOT NULL,
                        rationale TEXT NOT NULL
                    )
                    """,
                )
                return
            try:
                with self._connection() as conn:
                    column_sql = ",\n                ".join(
                        f"{name} {definition}" for name, definition in ANALYSIS_COLUMNS.items()
                    )
                    conn.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS analysis (
                            {column_sql}
                        )
                        """
                    )
                    existing_columns = {
                        row[1] for row in conn.execute("PRAGMA table_info(analysis)").fetchall()
                    }
                    for name, definition in ANALYSIS_COLUMNS.items():
                        if name not in existing_columns:
                            conn.execute(f"ALTER TABLE analysis ADD COLUMN {name} {definition}")
                    self._migrate_sector_names(conn)
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS portfolio_memberships (
                            ticker TEXT NOT NULL,
                            portfolio TEXT NOT NULL,
                            PRIMARY KEY (ticker, portfolio)
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS decision_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            portfolio TEXT NOT NULL,
                            ticker TEXT NOT NULL,
                            recommendation TEXT NOT NULL,
                            rationale TEXT NOT NULL
                        )
                        """
                    )
            except sqlite3.DatabaseError as exc:
                if self._storage_mode == "disk" and self._enable_in_memory_fallback(exc):
                    self.create_tables()
                    return
                raise

    @property
    def storage_notice(self):
        return self._fallback_notice

    @property
    def uses_persistent_storage(self):
        if self._backend == "postgres":
            return True
        return self._storage_mode == "disk" and self.db_path is not None

    @property
    def storage_label(self):
        if self._backend == "postgres":
            return self._redacted_postgres_label()
        if self.uses_persistent_storage:
            return str(self.db_path)
        return "In-memory session store"

    @property
    def storage_backend(self):
        return self._backend

    @property
    def supports_database_download(self):
        return self._backend == "sqlite" and self.uses_persistent_storage and self.db_path is not None

    @property
    def supports_portfolio_memberships(self):
        return self._backend != "postgres" or "portfolio_memberships" not in self._postgres_optional_table_errors

    @property
    def supports_decision_log(self):
        return self._backend != "postgres" or "decision_log" not in self._postgres_optional_table_errors

    def _postgres_column_definition(self, definition):
        mapping = {
            "TEXT PRIMARY KEY": "TEXT PRIMARY KEY",
            "TEXT": "TEXT",
            "REAL": "DOUBLE PRECISION",
            "INTEGER": "INTEGER",
        }
        return mapping[definition]

    def _redacted_postgres_label(self):
        try:
            parsed = urlparse(self._postgres_dsn or "")
        except ValueError:
            return "postgresql://configured"
        host = parsed.hostname or "unknown-host"
        port = parsed.port or 5432
        database = parsed.path.lstrip("/") or "unknown-db"
        user = parsed.username or "unknown-user"
        return f"postgresql://{user}@{host}:{port}/{database}"

    def _read_dataframe(self, conn, query, params=None):
        if self._backend == "postgres":
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                rows = cursor.fetchall()
                columns = [desc.name for desc in cursor.description] if cursor.description else []
            return pd.DataFrame(rows, columns=columns)
        return pd.read_sql_query(query, conn, params=params)
    def save_analysis(self, data):
        keys = list(data.keys())
        if self._backend == "postgres":
            placeholders = ", ".join(["%s"] * len(keys))
            columns = ", ".join(f'"{key}"' for key in keys)
            update_clause = ", ".join(
                f'"{key}"=EXCLUDED."{key}"' for key in keys if key != "Ticker"
            )
            sql = (
                f'INSERT INTO analysis ({columns}) VALUES ({placeholders}) '
                f'ON CONFLICT("Ticker") DO UPDATE SET {update_clause}'
            )
        else:
            placeholders = ", ".join(["?"] * len(keys))
            columns = ", ".join(keys)
            update_clause = ", ".join(
                f"{key}=excluded.{key}" for key in keys if key != "Ticker"
            )
            sql = (
                f"INSERT INTO analysis ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(Ticker) DO UPDATE SET {update_clause}"
            )
        with self._write_lock:
            try:
                with self._connection() as conn:
                    conn.execute(sql, list(data.values()))
            except (sqlite3.DatabaseError, psycopg.Error) as exc:
                if self._backend != "postgres" and self._storage_mode == "disk" and self._enable_in_memory_fallback(exc):
                    with self._connection() as conn:
                        conn.execute(sql, list(data.values()))
                    return
                raise

    def get_analysis(self, ticker):
        query = 'SELECT * FROM analysis WHERE "Ticker"=%s' if self._backend == "postgres" else "SELECT * FROM analysis WHERE Ticker=?"
        try:
            with self._connection() as conn:
                return self._read_dataframe(conn, query, params=(ticker,))
        except (pd.errors.DatabaseError, sqlite3.DatabaseError, psycopg.Error):
            self.create_tables()
            with self._connection() as conn:
                return self._read_dataframe(conn, query, params=(ticker,))

    def get_all_analyses(self):
        try:
            with self._connection() as conn:
                return self._read_dataframe(conn, "SELECT * FROM analysis")
        except (pd.errors.DatabaseError, sqlite3.DatabaseError, psycopg.Error):
            self.create_tables()
            with self._connection() as conn:
                return self._read_dataframe(conn, "SELECT * FROM analysis")

    def get_portfolio_memberships(self, portfolio=None):
        if self._backend == "postgres" and not self.supports_portfolio_memberships:
            return pd.DataFrame(columns=["ticker", "portfolio"])
        normalized_portfolio = str(portfolio or "").strip()
        ph = self._ph()
        query = "SELECT ticker, portfolio FROM portfolio_memberships"
        params = []
        if normalized_portfolio:
            query += f" WHERE portfolio={ph}"
            params.append(normalized_portfolio)
        query += " ORDER BY portfolio, ticker"

        try:
            with self._connection() as conn:
                return self._read_dataframe(conn, query, params=tuple(params))
        except (pd.errors.DatabaseError, sqlite3.DatabaseError, psycopg.Error):
            self.create_tables()
            if self._backend == "postgres" and not self.supports_portfolio_memberships:
                return pd.DataFrame(columns=["ticker", "portfolio"])
            with self._connection() as conn:
                return self._read_dataframe(conn, query, params=tuple(params))

    def get_portfolio_tickers(self, portfolio):
        memberships = self.get_portfolio_memberships(portfolio=portfolio)
        if memberships.empty or "ticker" not in memberships.columns:
            return []
        return (
            memberships["ticker"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .drop_duplicates()
            .tolist()
        )

    def save_portfolio_memberships(self, ticker, portfolios):
        normalized_ticker = normalize_ticker(ticker)
        normalized_portfolios = []
        seen = set()
        for portfolio_name in portfolios or []:
            cleaned = str(portfolio_name or "").strip()
            if cleaned and cleaned not in seen:
                normalized_portfolios.append(cleaned)
                seen.add(cleaned)
        if not normalized_ticker:
            return
        if self._backend == "postgres" and not self.supports_portfolio_memberships:
            return

        if self._backend == "postgres":
            delete_sql = "DELETE FROM portfolio_memberships WHERE ticker=%s"
            insert_sql = (
                "INSERT INTO portfolio_memberships (ticker, portfolio) VALUES (%s, %s) "
                "ON CONFLICT (ticker, portfolio) DO NOTHING"
            )
        else:
            delete_sql = "DELETE FROM portfolio_memberships WHERE ticker=?"
            insert_sql = (
                "INSERT OR IGNORE INTO portfolio_memberships (ticker, portfolio) VALUES (?, ?)"
            )

        with self._write_lock:
            try:
                with self._connection() as conn:
                    conn.execute(delete_sql, (normalized_ticker,))
                    for portfolio_name in normalized_portfolios:
                        conn.execute(insert_sql, (normalized_ticker, portfolio_name))
            except (sqlite3.DatabaseError, psycopg.Error):
                raise

    def add_decision_log_entry(self, portfolio, ticker, recommendation, rationale, timestamp=None):
        normalized_portfolio = str(portfolio or "").strip()
        normalized_ticker = normalize_ticker(ticker)
        normalized_recommendation = str(recommendation or "").strip().title()
        normalized_rationale = str(rationale or "").strip()
        timestamp_text = str(timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if not normalized_portfolio or not normalized_ticker or not normalized_recommendation or not normalized_rationale:
            return
        if self._backend == "postgres" and not self.supports_decision_log:
            return

        ph = self._ph()
        insert_sql = (
            f"INSERT INTO decision_log (timestamp, portfolio, ticker, recommendation, rationale) "
            f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph})"
        )

        with self._write_lock:
            try:
                with self._connection() as conn:
                    conn.execute(
                        insert_sql,
                        (
                            timestamp_text,
                            normalized_portfolio,
                            normalized_ticker,
                            normalized_recommendation,
                            normalized_rationale,
                        ),
                    )
            except (sqlite3.DatabaseError, psycopg.Error):
                raise

    def get_decision_log(self, portfolio=None, ticker=None):
        if self._backend == "postgres" and not self.supports_decision_log:
            return pd.DataFrame(columns=["id", "timestamp", "portfolio", "ticker", "recommendation", "rationale"])
        ph = self._ph()
        filters = []
        params = []
        if str(portfolio or "").strip():
            filters.append(f"portfolio={ph}")
            params.append(str(portfolio).strip())
        if str(ticker or "").strip():
            filters.append(f"ticker={ph}")
            params.append(normalize_ticker(ticker))

        query = "SELECT id, timestamp, portfolio, ticker, recommendation, rationale FROM decision_log"
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY timestamp DESC, id DESC"

        try:
            with self._connection() as conn:
                return self._read_dataframe(conn, query, params=tuple(params))
        except (pd.errors.DatabaseError, sqlite3.DatabaseError, psycopg.Error):
            self.create_tables()
            if self._backend == "postgres" and not self.supports_decision_log:
                return pd.DataFrame(columns=["id", "timestamp", "portfolio", "ticker", "recommendation", "rationale"])
            with self._connection() as conn:
                return self._read_dataframe(conn, query, params=tuple(params))


@st.cache_resource
def get_database_manager():
    return DatabaseManager(DATABASE_URL)


