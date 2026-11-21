from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def apply_migration(migration_path: Path) -> None:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    sql = migration_path.read_text()

    statements = [
        statement.strip() for statement in sql.split(";") if statement.strip()
    ]

    engine = create_async_engine(database_url)

    try:
        async with engine.begin() as connection:
            for statement in statements:
                await connection.execute(text(statement))
    finally:
        await engine.dispose()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python scripts/apply_sql_migration.py <migration.sql>"
        )

    migration_path = Path(sys.argv[1])

    if not migration_path.is_file():
        raise SystemExit(f"Migration file not found: {migration_path}")

    asyncio.run(apply_migration(migration_path))


if __name__ == "__main__":
    main()
