"""Migration versioning: the Alembic chain must build exactly the ORM schema and be fully reversible."""

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from quantpulse.db import migrate
from quantpulse.db.base import Base

EXPECTED_CHAIN = ["0001", "0002", "0003", "0004", "0005", "0006"]


def _url(tmp_path):
    return f"sqlite+aiosqlite:///{tmp_path / 'm.db'}"


def test_revision_chain_is_linear():
    script = ScriptDirectory.from_config(migrate.alembic_config("sqlite://"))
    revs = [r.revision for r in reversed(list(script.walk_revisions()))]
    assert revs == EXPECTED_CHAIN
    assert migrate.head_revision() == "0006"


def test_upgrade_matches_models_and_downgrade_is_clean(tmp_path):
    url = _url(tmp_path)
    migrate.upgrade(url)
    assert migrate.current_revision(url) == "0006"
    engine = create_engine(migrate.sync_url(url))
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"models and migrations diverged: {diff}"
    assert set(inspect(engine).get_table_names()) == set(Base.metadata.tables) | {"alembic_version"}
    engine.dispose()

    migrate.downgrade(url, "base")
    engine = create_engine(migrate.sync_url(url))
    assert inspect(engine).get_table_names() == ["alembic_version"]
    engine.dispose()


def test_stepwise_upgrade_and_downgrade(tmp_path):
    url = _url(tmp_path)
    for rev in EXPECTED_CHAIN:
        migrate.upgrade(url, rev)
        assert migrate.current_revision(url) == rev
    for rev in reversed(EXPECTED_CHAIN[:-1]):
        migrate.downgrade(url, rev)
        assert migrate.current_revision(url) == rev
    migrate.downgrade(url, "base")
    assert migrate.current_revision(url) is None
