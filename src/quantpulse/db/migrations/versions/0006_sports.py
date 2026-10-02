"""Sports games and Elo team ratings.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sports_games",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("event_id", sa.String(length=24), nullable=False),
        sa.Column("league", sa.String(length=24), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("season_type", sa.Integer(), nullable=False),
        sa.Column("week", sa.Integer(), nullable=True),
        sa.Column("start_time", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("home_team_id", sa.String(length=16), nullable=False),
        sa.Column("away_team_id", sa.String(length=16), nullable=False),
        sa.Column("home_name", sa.String(length=80), nullable=False),
        sa.Column("away_name", sa.String(length=80), nullable=False),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("state", sa.String(length=8), nullable=False),
        sa.Column("completed", sa.Boolean(), nullable=False),
        sa.Column("neutral_site", sa.Boolean(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sports_games")),
        sa.UniqueConstraint("event_id", name=op.f("uq_sports_games_event_id")),
    )
    op.create_index("ix_sports_games_league_season", "sports_games", ["league", "season"])

    op.create_table(
        "team_ratings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("league", sa.String(length=24), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.String(length=16), nullable=False),
        sa.Column("team_name", sa.String(length=80), nullable=False),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("games", sa.Integer(), nullable=False),
        sa.Column("wins", sa.Integer(), nullable=False),
        sa.Column("losses", sa.Integer(), nullable=False),
        sa.Column("ties", sa.Integer(), nullable=False),
        sa.Column("computed_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_team_ratings")),
        sa.UniqueConstraint(
            "league", "season", "team_id", name=op.f("uq_team_ratings_league_season_team_id")
        ),
    )


def downgrade() -> None:
    op.drop_table("team_ratings")
    op.drop_index("ix_sports_games_league_season", table_name="sports_games")
    op.drop_table("sports_games")
