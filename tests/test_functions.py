"""Tests for cross-dialect date_trunc SQL function compilers."""

import pytest
from sqlalchemy import column
from sqlalchemy.dialects import mysql, oracle, postgresql, sqlite
from sqlalchemy.dialects.mssql import base as mssql_base

from llmwatch.databases.functions import date_trunc


def _compile(expr, dialect):
    """Render a SQLAlchemy expression to a SQL string for the given dialect."""
    return expr.compile(dialect=dialect, compile_kwargs={"literal_binds": True}).string


# * Shared column used across all dialect tests
_col = column("created_at")


class TestDateTruncPostgreSQL:
    _dialect = postgresql.dialect()

    def test_hour_precision(self):
        sql = _compile(date_trunc("hour", _col), self._dialect)
        assert sql == "DATE_TRUNC('hour', created_at)"

    def test_day_precision(self):
        sql = _compile(date_trunc("day", _col), self._dialect)
        assert sql == "DATE_TRUNC('day', created_at)"

    def test_week_precision(self):
        sql = _compile(date_trunc("week", _col), self._dialect)
        assert sql == "DATE_TRUNC('week', created_at)"

    def test_month_precision(self):
        sql = _compile(date_trunc("month", _col), self._dialect)
        assert sql == "DATE_TRUNC('month', created_at)"

    def test_custom_precision_passes_through(self):
        # PostgreSQL compiler forwards any precision string without validation
        sql = _compile(date_trunc("year", _col), self._dialect)
        assert sql == "DATE_TRUNC('year', created_at)"


class TestDateTruncSQLite:
    _dialect = sqlite.dialect()

    def test_hour_precision(self):
        sql = _compile(date_trunc("hour", _col), self._dialect)
        assert sql == "STRFTIME('%Y-%m-%d %H:00:00', created_at)"

    def test_day_precision(self):
        sql = _compile(date_trunc("day", _col), self._dialect)
        assert sql == "STRFTIME('%Y-%m-%d 00:00:00', created_at)"

    def test_month_precision(self):
        sql = _compile(date_trunc("month", _col), self._dialect)
        assert sql == "STRFTIME('%Y-%m-01 00:00:00', created_at)"

    def test_week_precision_uses_weekday_offset(self):
        # ^ SQLite has no native week-trunc; emulated via STRFTIME('%w') subtraction
        sql = _compile(date_trunc("week", _col), self._dialect)
        assert "STRFTIME('%Y-%m-%d 00:00:00'" in sql
        assert "STRFTIME('%w'" in sql
        assert "days" in sql

    def test_unsupported_precision_raises(self):
        with pytest.raises(NotImplementedError, match="SQLite date_trunc does not support"):
            _compile(date_trunc("year", _col), self._dialect)


class TestDateTruncMySQL:
    _dialect = mysql.dialect()

    def test_hour_precision(self):
        sql = _compile(date_trunc("hour", _col), self._dialect)
        assert sql == "DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00')"

    def test_day_precision(self):
        sql = _compile(date_trunc("day", _col), self._dialect)
        assert sql == "DATE_FORMAT(created_at, '%Y-%m-%d 00:00:00')"

    def test_week_precision(self):
        sql = _compile(date_trunc("week", _col), self._dialect)
        assert sql == "DATE_FORMAT(created_at, '%Y-%u')"

    def test_month_precision(self):
        sql = _compile(date_trunc("month", _col), self._dialect)
        assert sql == "DATE_FORMAT(created_at, '%Y-%m-01 00:00:00')"

    def test_unsupported_precision_raises(self):
        with pytest.raises(NotImplementedError, match="MySQL date_trunc does not support"):
            _compile(date_trunc("year", _col), self._dialect)


class TestDateTruncOracle:
    _dialect = oracle.dialect()

    def test_hour_precision(self):
        sql = _compile(date_trunc("hour", _col), self._dialect)
        assert sql == "TRUNC(created_at, 'HH24')"

    def test_day_precision(self):
        sql = _compile(date_trunc("day", _col), self._dialect)
        assert sql == "TRUNC(created_at, 'DD')"

    def test_week_precision(self):
        sql = _compile(date_trunc("week", _col), self._dialect)
        assert sql == "TRUNC(created_at, 'IW')"

    def test_month_precision(self):
        sql = _compile(date_trunc("month", _col), self._dialect)
        assert sql == "TRUNC(created_at, 'MONTH')"

    def test_unsupported_precision_raises(self):
        with pytest.raises(NotImplementedError, match="Oracle date_trunc does not support"):
            _compile(date_trunc("year", _col), self._dialect)


class TestDateTruncMSSQL:
    _dialect = mssql_base.MSDialect()

    def test_hour_precision(self):
        sql = _compile(date_trunc("hour", _col), self._dialect)
        assert sql == "DATETRUNC(hour, created_at)"

    def test_day_precision(self):
        sql = _compile(date_trunc("day", _col), self._dialect)
        assert sql == "DATETRUNC(day, created_at)"

    def test_week_precision(self):
        sql = _compile(date_trunc("week", _col), self._dialect)
        assert sql == "DATETRUNC(week, created_at)"

    def test_month_precision(self):
        sql = _compile(date_trunc("month", _col), self._dialect)
        assert sql == "DATETRUNC(month, created_at)"


class TestDateTruncDefaultDialect:
    """The default (catch-all) compiler raises NotImplementedError for unknown dialects."""

    def test_unknown_dialect_raises(self):
        # ^ DuckDB is not registered, so it falls through to the default @compiles handler
        from sqlalchemy.engine import default as sa_default

        unknown_dialect = sa_default.DefaultDialect()
        unknown_dialect.name = "duckdb"

        with pytest.raises(NotImplementedError, match="date_trunc is not implemented for dialect"):
            _compile(date_trunc("day", _col), unknown_dialect)


class TestDateTruncInheritCache:
    """Verify the inherit_cache flag required by SQLAlchemy's caching mechanism."""

    def test_inherit_cache_is_true(self):
        assert date_trunc.inherit_cache is True
