"""Companies, normalised annual statements, SEC filings and analyst-estimate snapshots.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=True),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("shares_outstanding", sa.Float(), nullable=True),
        sa.Column("shares_as_of", sa.Date(), nullable=True),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_companies")),
        sa.UniqueConstraint("symbol", name=op.f("uq_companies_symbol")),
    )
    op.create_index(op.f("ix_companies_cik"), "companies", ["cik"])

    op.create_table(
        "financial_statements",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("form", sa.String(length=12), nullable=False),
        sa.Column("filed", sa.Date(), nullable=True),
        sa.Column("accession", sa.String(length=25), nullable=True),
        sa.Column("revenue", sa.Float(), nullable=True),
        sa.Column("gross_profit", sa.Float(), nullable=True),
        sa.Column("operating_income", sa.Float(), nullable=True),
        sa.Column("net_income", sa.Float(), nullable=True),
        sa.Column("pretax_income", sa.Float(), nullable=True),
        sa.Column("income_tax", sa.Float(), nullable=True),
        sa.Column("interest_expense", sa.Float(), nullable=True),
        sa.Column("depreciation_amortization", sa.Float(), nullable=True),
        sa.Column("total_assets", sa.Float(), nullable=True),
        sa.Column("total_liabilities", sa.Float(), nullable=True),
        sa.Column("stockholders_equity", sa.Float(), nullable=True),
        sa.Column("cash", sa.Float(), nullable=True),
        sa.Column("total_debt", sa.Float(), nullable=True),
        sa.Column("current_assets", sa.Float(), nullable=True),
        sa.Column("current_liabilities", sa.Float(), nullable=True),
        sa.Column("operating_cash_flow", sa.Float(), nullable=True),
        sa.Column("capital_expenditure", sa.Float(), nullable=True),
        sa.Column("diluted_eps", sa.Float(), nullable=True),
        sa.Column("diluted_shares", sa.Float(), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("ingested_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_financial_statements")),
        sa.UniqueConstraint("symbol", "fiscal_year", name=op.f("uq_financial_statements_symbol_fiscal_year")),
    )
    op.create_index(op.f("ix_financial_statements_symbol"), "financial_statements", ["symbol"])

    op.create_table(
        "sec_filings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("accession", sa.String(length=25), nullable=False),
        sa.Column("form", sa.String(length=16), nullable=False),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=True),
        sa.Column("primary_document", sa.String(length=200), nullable=True),
        sa.Column("url", sa.String(length=400), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sec_filings")),
        sa.UniqueConstraint("accession", name=op.f("uq_sec_filings_accession")),
    )
    op.create_index(op.f("ix_sec_filings_symbol"), "sec_filings", ["symbol"])

    op.create_table(
        "analyst_estimate_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("captured_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analyst_estimate_snapshots")),
    )
    op.create_index(
        "ix_analyst_estimate_snapshots_symbol_captured_at",
        "analyst_estimate_snapshots",
        ["symbol", "captured_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_analyst_estimate_snapshots_symbol_captured_at", table_name="analyst_estimate_snapshots")
    op.drop_table("analyst_estimate_snapshots")
    op.drop_index(op.f("ix_sec_filings_symbol"), table_name="sec_filings")
    op.drop_table("sec_filings")
    op.drop_index(op.f("ix_financial_statements_symbol"), table_name="financial_statements")
    op.drop_table("financial_statements")
    op.drop_index(op.f("ix_companies_cik"), table_name="companies")
    op.drop_table("companies")
