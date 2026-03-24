"""Cross-dialect SQL functions using SQLAlchemy's @compiles."""

from typing import Any

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.expression import FunctionElement
from sqlalchemy.types import DateTime

# * SQLite STRFTIME format mapping per precision
_SQLITE_FORMATS: dict[str, str] = {
    "hour": "%Y-%m-%d %H:00:00",
    "day": "%Y-%m-%d 00:00:00",
    "month": "%Y-%m-01 00:00:00",
}

# * MySQL DATE_FORMAT format mapping per precision
_MYSQL_FORMATS: dict[str, str] = {
    "hour": "%Y-%m-%d %H:00:00",
    "day": "%Y-%m-%d 00:00:00",
    "week": "%Y-%u",
    "month": "%Y-%m-01 00:00:00",
}

# * Oracle TRUNC fmt mapping per precision
_ORACLE_FORMATS: dict[str, str] = {
    "hour": "HH24",
    "day": "DD",
    "week": "IW",
    "month": "MONTH",
}


class date_trunc(FunctionElement[Any]):
    """Portable date truncation across SQLite, PostgreSQL, MySQL, Oracle, MSSQL."""

    type = DateTime()
    inherit_cache = True

    def __init__(self, precision: str, col: object) -> None:
        self.precision = precision
        super().__init__(col)


@compiles(date_trunc, "postgresql")
def _date_trunc_postgresql(element: date_trunc, compiler: SQLCompiler, **kw: Any) -> str:
    col = compiler.process(list(element.clauses)[0], **kw)
    return f"DATE_TRUNC('{element.precision}', {col})"


@compiles(date_trunc, "sqlite")
def _date_trunc_sqlite(element: date_trunc, compiler: SQLCompiler, **kw: Any) -> str:
    col = compiler.process(list(element.clauses)[0], **kw)
    precision = element.precision
    if precision == "week":
        # ^ SQLite has no native week-trunc; subtract weekday offset to get Monday
        return f"STRFTIME('%Y-%m-%d 00:00:00', {col}, '-' || CAST(STRFTIME('%w', {col}) AS INTEGER) || ' days')"
    fmt = _SQLITE_FORMATS.get(precision)
    if fmt is None:
        raise NotImplementedError(f"SQLite date_trunc does not support precision: {precision!r}")
    return f"STRFTIME('{fmt}', {col})"


@compiles(date_trunc, "mysql")
def _date_trunc_mysql(element: date_trunc, compiler: SQLCompiler, **kw: Any) -> str:
    col = compiler.process(list(element.clauses)[0], **kw)
    fmt = _MYSQL_FORMATS.get(element.precision)
    if fmt is None:
        raise NotImplementedError(
            f"MySQL date_trunc does not support precision: {element.precision!r}"
        )
    return f"DATE_FORMAT({col}, '{fmt}')"


@compiles(date_trunc, "oracle")
def _date_trunc_oracle(element: date_trunc, compiler: SQLCompiler, **kw: Any) -> str:
    col = compiler.process(list(element.clauses)[0], **kw)
    oracle_fmt = _ORACLE_FORMATS.get(element.precision)
    if oracle_fmt is None:
        raise NotImplementedError(
            f"Oracle date_trunc does not support precision: {element.precision!r}"
        )
    return f"TRUNC({col}, '{oracle_fmt}')"


@compiles(date_trunc, "mssql")
def _date_trunc_mssql(element: date_trunc, compiler: SQLCompiler, **kw: Any) -> str:
    col = compiler.process(list(element.clauses)[0], **kw)
    return f"DATETRUNC({element.precision}, {col})"


@compiles(date_trunc)
def _date_trunc_default(element: date_trunc, compiler: SQLCompiler, **kw: Any) -> str:
    raise NotImplementedError(
        f"date_trunc is not implemented for dialect: {compiler.dialect.name!r}"
    )
