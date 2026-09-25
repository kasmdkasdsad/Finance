"""Asset Lifecycle: 2025 Hyundai Elantra Limited operating costs, depreciation and maintenance."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, money, num, status_badge

STATUS_LABEL = {
    "overdue": ":material/error: Overdue",
    "due_soon": ":material/schedule: Due soon",
    "ok": ":material/check_circle: OK",
}


def _create_form() -> None:
    with st.form("new_vehicle"):
        st.markdown("**Register your vehicle** (profile: 2025 Hyundai Elantra Limited)")
        c = st.columns(3)
        nickname = c[0].text_input("Nickname", "My Elantra")
        purchase_date = c[1].date_input("Purchase date", date(2025, 6, 1))
        price = c[2].number_input("Purchase price ($, 0 = MSRP + destination)", 0.0, 200000.0, 0.0, 100.0)
        o = st.columns(3)
        miles = o[0].number_input("Miles per year", 1000, 100000, 12000, 500)
        city = o[1].slider("City driving share", 0.0, 1.0, 0.55, 0.05)
        regions = guarded(lambda: api().get("/fuel/regions"), "regions") or {"NUS": "U.S. average"}
        region = o[2].selectbox(
            "Fuel price region (EIA)", list(regions), format_func=lambda k: f"{regions[k]} ({k})"
        )
        if st.form_submit_button("Create", type="primary"):
            payload = {
                "nickname": nickname,
                "purchase_date": purchase_date.isoformat(),
                "annual_miles": miles,
                "city_share": city,
                "fuel_region": region,
            }
            if price > 0:
                payload["purchase_price"] = price
            if guarded(lambda: api().post("/vehicles", payload), "create vehicle"):
                st.rerun()


def _log_forms(vid: int, codes: list[str], odometer: float) -> None:
    a, b, c = st.tabs(["Fuel fill-up", "Odometer / telemetry", "Service record"])
    with a, st.form("fuel"):
        x = st.columns(4)
        odo = x[0].number_input("Odometer", 0.0, 2e6, float(odometer), 1.0)
        gal = x[1].number_input("Gallons", 0.01, 30.0, 10.0, 0.01)
        ppg = x[2].number_input("$ / gallon", 0.01, 20.0, 3.19, 0.01)
        full = x[3].checkbox("Full tank", True)
        if st.form_submit_button("Log fill-up"):
            payload = {
                "filled_at": datetime.now(UTC).isoformat(),
                "odometer": odo,
                "gallons": gal,
                "price_per_gallon": ppg,
                "full_tank": full,
            }
            if guarded(lambda: api().post(f"/vehicles/{vid}/fuel-logs", payload), "fuel log"):
                st.rerun()
    with b, st.form("telemetry"):
        x = st.columns(2)
        odo = x[0].number_input("Odometer reading", 0.0, 2e6, float(odometer), 1.0)
        fuel = x[1].slider("Fuel level %", 0, 100, 50)
        if st.form_submit_button("Record"):
            payload = {"recorded_at": datetime.now(UTC).isoformat(), "odometer": odo, "fuel_level_pct": fuel}
            if guarded(lambda: api().post(f"/vehicles/{vid}/telemetry", payload), "telemetry"):
                st.rerun()
    with c, st.form("service"):
        x = st.columns(4)
        code = x[0].selectbox("Service", codes)
        on = x[1].date_input("Performed on", date.today())
        odo = x[2].number_input("Odometer at service", 0.0, 2e6, float(odometer), 1.0)
        cost = x[3].number_input("Cost $", 0.0, 10000.0, 0.0, 1.0)
        if st.form_submit_button("Add record"):
            payload = {"service_code": code, "performed_on": on.isoformat(), "odometer": odo, "cost": cost}
            if guarded(lambda: api().post(f"/vehicles/{vid}/maintenance", payload), "maintenance"):
                st.rerun()


def render() -> None:
    st.title("Asset Lifecycle · 2025 Hyundai Elantra Limited")
    vehicles = guarded(lambda: api().get("/vehicles"), "vehicles")
    if vehicles is None:
        return
    if not vehicles:
        _create_form()
        return
    pick = st.selectbox("Vehicle", vehicles, format_func=lambda v: f"{v['nickname']} (#{v['id']})")
    res = guarded(lambda: api().get(f"/vehicles/{pick['id']}/dashboard"), "dashboard")
    if not res:
        return
    d = res["data"]
    composite_badges(res["meta"])
    if d["telemetry_source"] == "synthetic":
        st.badge("Telemetry SIMULATED — log odometer or fill-ups", icon=":material/science:", color="red")
    for w in d["warnings"]:
        st.caption(f":material/info: {w}")

    k = st.columns(5)
    k[0].metric("Odometer", f"{num(d['odometer'], 0)} mi", help=f"{num(d['avg_daily_miles'], 1)} mi/day")
    k[1].metric(
        "Current value",
        money(d["current_value"]),
        f"-{money(d['total_depreciation'])} since purchase",
        delta_color="off",
    )
    k[2].metric("Cost per mile", f"${d['cost_per_mile']['total']:.3f}")
    k[3].metric(
        "Fuel economy",
        f"{num(d['effective_mpg'], 1)} mpg",
        help="Realized from fill-ups" if d["realized_mpg"] else "EPA, adjusted to your mix",
    )
    fp = d["fuel_price"]
    k[4].metric(
        f"{fp['grade'].title()} gas · {fp['region']}",
        f"${fp['price']:.3f}/gal",
        help=f"EIA week of {fp['period']}",
    )

    left, right = st.columns(2, gap="large")
    with left:
        cpm = d["cost_per_mile"]
        parts = [
            ("Fuel", cpm["fuel"]),
            ("Depreciation", cpm["depreciation"]),
            ("Maintenance", cpm["maintenance"]),
        ]
        fig = go.Figure()
        for i, (label, value) in enumerate(parts):
            fig.add_trace(
                go.Bar(
                    y=["Cost per mile"],
                    x=[value * 100],
                    name=label,
                    orientation="h",
                    marker={
                        "color": charts.series(i),
                        "line": {"color": charts.theme()["surface"], "width": 2},
                    },
                    hovertemplate=label + ": %{x:.1f}¢/mi<extra></extra>",
                )
            )
        charts.base_layout(fig, "Cost per mile breakdown (cents)", height=200, barmode="stack")
        fig.update_xaxes(ticksuffix="¢", showgrid=True, gridcolor=charts.theme()["grid"])
        charts.show(fig)
        proj = pd.DataFrame({"monthly": d["projected_monthly_cost"], "annual": d["projected_annual_cost"]})
        st.dataframe(proj.map(lambda v: f"${v:,.2f}"), width="stretch")
    with right:
        curve = pd.DataFrame(d["depreciation_curve"])
        fig = go.Figure(
            go.Scatter(
                x=curve["on"],
                y=curve["value"],
                mode="lines",
                line={"color": charts.series(0), "width": 2},
                name="Modelled value",
                hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[date.today().isoformat()],
                y=[d["current_value"]],
                mode="markers+text",
                text=["today"],
                textposition="top right",
                marker={
                    "size": 10,
                    "color": charts.series(1),
                    "line": {"width": 2, "color": charts.theme()["surface"]},
                },
                name="Today",
            )
        )
        charts.base_layout(
            fig, "Depreciation curve (10 years, your mileage)", height=300, hovermode="x unified"
        )
        fig.update_yaxes(tickprefix="$")
        charts.show(fig)
        st.caption(d["profile"]["depreciation"]["source"])

    st.subheader("Maintenance schedule")
    st.caption(d["profile"]["maintenance_source"])
    maint = pd.DataFrame(d["maintenance"])
    maint["status"] = maint["status"].map(STATUS_LABEL)
    st.dataframe(
        maint[
            [
                "status",
                "name",
                "next_due_odometer",
                "miles_remaining",
                "next_due_date",
                "projected_due_date",
                "estimated_cost",
                "last_service_odometer",
                "last_service_date",
            ]
        ],
        hide_index=True,
    )

    fuel = guarded(lambda: api().get("/fuel/prices", region=fp["region"], grade=fp["grade"]), "fuel history")
    if fuel:
        hist = pd.DataFrame(fuel["data"]["history"])
        fig = go.Figure(
            go.Scatter(
                x=hist["period"],
                y=hist["price"],
                mode="lines",
                line={"color": charts.series(0), "width": 2},
                name="Price",
                hovertemplate="%{x}: $%{y:.3f}<extra></extra>",
            )
        )
        charts.base_layout(
            fig,
            f"Weekly {fuel['data']['grade']} retail price · {fuel['data']['region_name']}",
            height=260,
            showlegend=False,
            hovermode="x unified",
        )
        fig.update_yaxes(tickprefix="$")
        charts.show(fig)
        status_badge(fuel["meta"], label="EIA")

    st.subheader("Log activity")
    _log_forms(pick["id"], [m["code"] for m in d["maintenance"]], d["odometer"])
    with st.expander("Specifications & EPA data"):
        status_badge(res["meta"]["sources"]["epa"], label="fueleconomy.gov")
        p = d["profile"]
        st.json(
            {
                "engine": p["engine"],
                "transmission": p["transmission"],
                "horsepower": p["horsepower"],
                "torque_lb_ft": p["torque_lb_ft"],
                "tank_gallons": p["tank_gallons"],
                "epa": d["epa_live"],
                "pricing": p["pricing"],
            }
        )
