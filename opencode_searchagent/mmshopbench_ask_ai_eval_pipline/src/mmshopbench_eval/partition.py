

from __future__ import annotations

import datetime as dt
from typing import Any


def normalize_partition_spec(partition: str) -> str:
    value = str(partition or "").strip()
    if not value:
        return f"ds={dt.datetime.now().strftime('%Y%m%d')}"
    if "=" not in value:
        return f"ds={value}"
    return value


def partition_value(partition_spec: str) -> str:
    if "=" in partition_spec:
        return partition_spec.split("=", 1)[1].strip("'\"")
    return partition_spec


def sql_quote(value: Any) -> str:
    return "'" + str(value or "").replace("'", "''") + "'"


def partition_spec_to_where(partition: str) -> str:
    partition = normalize_partition_spec(str(partition or "").strip())
    if not partition:
        return ""
    clauses: list[str] = []
    for part in partition.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            piece = f"ds={piece}"
        key, value = piece.split("=", 1)
        value = value.strip().strip('"').strip("'")
        clauses.append(f"{key.strip()} = {sql_quote(value)}")
    return " AND ".join(clauses)


def build_select_sql(table_name: str, partition: str = "", extra_where: str = "", limit: int = 0) -> str:
    clauses: list[str] = []
    partition_where = partition_spec_to_where(partition)
    if partition_where:
        clauses.append(partition_where)
    if extra_where:
        clauses.append(extra_where)
    sql = f"SELECT * FROM {table_name}"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if limit and limit > 0:
        sql += f" LIMIT {limit}"
    return sql
