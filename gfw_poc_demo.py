"""
AIS PoC extension focused on:
1) Ban-compliance index (June-Aug vs non-ban months)
2) In-zone hotspot persistence (one-off vs recurring vs chronic)

Run:
  python gfw_poc_compliance_persistence.py
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import HeatMap

from gfw_poc_final import (
    ARTISANAL_NM,
    BAN_MONTHS,
    BBOX,
    build_artisanal_zone,
    build_monthly_summary,
    classify_in_zone,
    fetch_monthly_effort,
)

# Non-GUI backend (safe for headless environments)
mpl.use("Agg")

OUT_DIR = Path(__file__).parent
OUT_DASHBOARD = OUT_DIR / "poc_ais_compliance_persistence_dashboard_tn_puducherry.png"
OUT_HTML = OUT_DIR / "poc_ais_compliance_persistence_tn_puducherry.html"
OUT_MONTHLY_CSV = OUT_DIR / "poc_monthly_summary_with_index_tn_puducherry.csv"
OUT_COMPLIANCE_CSV = OUT_DIR / "poc_ban_compliance_index_tn_puducherry.csv"
OUT_PERSISTENCE_CSV = OUT_DIR / "poc_hotspot_persistence_tn_puducherry.csv"


def _season_label(dt: pd.Timestamp) -> str:
    month = int(dt.month)
    if month in BAN_MONTHS:
        return "Ban window (Jun-Aug)"
    if month in {10, 11, 12}:
        return "NE monsoon"
    if month in {3, 4, 5}:
        return "Pre-monsoon"
    return "Other months"


def _compliance_index(ban_share_pct: float, non_ban_share_pct: float) -> float:
    if pd.isna(non_ban_share_pct) or non_ban_share_pct <= 0:
        return 0.0
    return 100.0 * (1.0 - (ban_share_pct / non_ban_share_pct))


def build_ban_compliance_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    def summarize(df: pd.DataFrame, label: str) -> dict:
        ban = df[df["is_ban_month"]]
        non_ban = df[~df["is_ban_month"]]

        ban_inside = float(ban["inside_hours"].sum())
        ban_total = float(ban["total_hours"].sum())
        non_inside = float(non_ban["inside_hours"].sum())
        non_total = float(non_ban["total_hours"].sum())

        ban_share = 100.0 * ban_inside / ban_total if ban_total > 0 else 0.0
        non_share = 100.0 * non_inside / non_total if non_total > 0 else 0.0
        ratio = ban_share / non_share if non_share > 0 else 0.0

        return {
            "scope": label,
            "ban_inside_hours": ban_inside,
            "ban_total_hours": ban_total,
            "ban_inside_share_pct": ban_share,
            "non_ban_inside_hours": non_inside,
            "non_ban_total_hours": non_total,
            "non_ban_inside_share_pct": non_share,
            "ban_to_non_ban_inside_share_ratio": ratio,
            "compliance_index_pct": _compliance_index(ban_share, non_share),
        }

    rows = [summarize(monthly, "overall")]
    for year in sorted(monthly["period_dt"].dt.year.unique().tolist()):
        rows.append(summarize(monthly[monthly["period_dt"].dt.year == year], str(year)))
    return pd.DataFrame(rows)


def build_hotspot_persistence(classified: pd.DataFrame) -> pd.DataFrame:
    inside = classified[
        classified["inside_artisanal_zone"] & (classified["hours"] > 0)
    ].copy()
    if inside.empty:
        return pd.DataFrame(
            columns=[
                "lat",
                "lon",
                "active_months",
                "active_seasons",
                "ban_month_hits",
                "non_ban_month_hits",
                "total_hours",
                "mean_active_month_hours",
                "first_period",
                "last_period",
                "persistence_class",
            ]
        )

    if "period_dt" not in inside.columns:
        inside["period_dt"] = pd.to_datetime(inside["period"] + "-01")
    inside["is_ban_month"] = inside["period_dt"].dt.month.isin(BAN_MONTHS)
    inside["season"] = inside["period_dt"].apply(_season_label)
    agg = (
        inside.groupby(["lat", "lon"], as_index=False)
        .agg(
            active_months=("period", "nunique"),
            active_seasons=("season", "nunique"),
            ban_month_hits=("is_ban_month", "sum"),
            total_hours=("hours", "sum"),
            mean_active_month_hours=("hours", "mean"),
            first_period=("period", "min"),
            last_period=("period", "max"),
        )
        .sort_values(["active_months", "total_hours"], ascending=[False, False])
    )
    agg["non_ban_month_hits"] = agg["active_months"] - agg["ban_month_hits"]

    bins = [0, 1, 3, 8, 100]
    labels = ["one-off", "recurring (2-3)", "persistent (4-8)", "chronic (9+)"]
    agg["persistence_class"] = pd.cut(
        agg["active_months"], bins=bins, labels=labels, right=True
    ).astype(str)

    cols = [
        "lat",
        "lon",
        "active_months",
        "active_seasons",
        "ban_month_hits",
        "non_ban_month_hits",
        "total_hours",
        "mean_active_month_hours",
        "first_period",
        "last_period",
        "persistence_class",
    ]
    return agg[cols].reset_index(drop=True)


def save_tables(
    monthly: pd.DataFrame, compliance: pd.DataFrame, persistence: pd.DataFrame
) -> None:
    monthly.to_csv(OUT_MONTHLY_CSV, index=False)
    compliance.to_csv(OUT_COMPLIANCE_CSV, index=False)
    persistence.to_csv(OUT_PERSISTENCE_CSV, index=False)
    print(f"Saved -> {OUT_MONTHLY_CSV}")
    print(f"Saved -> {OUT_COMPLIANCE_CSV}")
    print(f"Saved -> {OUT_PERSISTENCE_CSV}")


def plot_dashboard(
    classified: pd.DataFrame,
    boundary: gpd.GeoSeries,
    zone: gpd.GeoSeries,
    monthly: pd.DataFrame,
    compliance: pd.DataFrame,
    persistence: pd.DataFrame,
) -> None:
    cells = classified.groupby(["lat", "lon"], as_index=False)["hours"].sum()
    if cells.empty:
        raise SystemExit("No non-zero effort cells available for dashboard plotting.")

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])
    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax_month = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[1, 1])

    ax_map.set_extent((BBOX[0], BBOX[2], BBOX[1], BBOX[3]), crs=ccrs.PlateCarree())
    ax_map.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "10m"),
        facecolor="#f5f9ff",
        zorder=1,
    )
    ax_map.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "10m", edgecolor="black", facecolor="#dcdcdc"
        ),
        zorder=2,
    )

    for geom in zone:
        if geom is None or geom.is_empty:
            continue
        ax_map.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor="cyan",
            edgecolor="none",
            alpha=0.12,
            zorder=3,
        )

    norm = mcolors.LogNorm(
        vmin=max(cells["hours"].quantile(0.05), 0.1),
        vmax=cells["hours"].quantile(0.99),
    )
    sc = ax_map.scatter(
        cells["lon"],
        cells["lat"],
        c=cells["hours"],
        cmap="hot_r",
        norm=norm,
        s=14,
        alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=6,
        linewidths=0,
    )

    for geom in boundary:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for part in parts:
            x, y = part.xy
            ax_map.plot(
                x,
                y,
                color="cyan",
                linewidth=1.3,
                transform=ccrs.PlateCarree(),
                zorder=7,
            )

    top_persistent = persistence.sort_values(
        ["active_months", "total_hours"], ascending=[False, False]
    ).head(15)
    if not top_persistent.empty:
        ax_map.scatter(
            top_persistent["lon"],
            top_persistent["lat"],
            s=30 + (top_persistent["active_months"] * 7),
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            transform=ccrs.PlateCarree(),
            zorder=8,
        )

    ax_map.add_feature(
        cfeature.NaturalEarthFeature("physical", "coastline", "10m"),
        linewidth=0.6,
        zorder=4,
    )
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax_map.set_title("AIS effort + 3 nm zone + top persistent in-zone hotspots")
    cbar = plt.colorbar(sc, ax=ax_map, shrink=0.65, pad=0.03)
    cbar.set_label("Fishing hours (log scale)")

    ax_month.plot(
        monthly["period_dt"],
        monthly["inside_share_pct"],
        color="#d7301f",
        linewidth=1.8,
        label="Inside-zone share (%)",
    )
    for _, row in monthly[monthly["is_ban_month"]].iterrows():
        start = row["period_dt"]
        end = start + pd.offsets.MonthEnd(1)
        ax_month.axvspan(start, end, color="#fee391", alpha=0.25, linewidth=0)
    ax_month.set_title("Inside-zone share over time (yellow = ban months)")
    ax_month.set_ylabel("Inside-zone share (%)")
    ax_month.grid(alpha=0.25, linestyle="--")
    ax_month.legend(frameon=False)

    comp_year = compliance[compliance["scope"] != "overall"].copy()
    comp_year["scope"] = comp_year["scope"].astype(str)
    colors = [
        "#2ca25f" if v >= 0 else "#de2d26"
        for v in comp_year["compliance_index_pct"].tolist()
    ]
    ax_comp.bar(comp_year["scope"], comp_year["compliance_index_pct"], color=colors)
    ax_comp.axhline(0, color="black", linewidth=0.9)
    ax_comp.set_ylabel("Compliance index (%)")
    ax_comp.set_title("Ban compliance index by year")
    ax_comp.grid(alpha=0.25, linestyle="--", axis="y")

    overall = compliance[compliance["scope"] == "overall"].iloc[0]
    pclass = persistence["persistence_class"].value_counts()
    summary = (
        f"Overall compliance index: {overall['compliance_index_pct']:.1f}%\n"
        f"Ban/non-ban share ratio: {overall['ban_to_non_ban_inside_share_ratio']:.2f}\n"
        f"Hotspots: chronic={int(pclass.get('chronic (9+)', 0))}, "
        f"persistent={int(pclass.get('persistent (4-8)', 0))}, "
        f"recurring={int(pclass.get('recurring (2-3)', 0))}, "
        f"one-off={int(pclass.get('one-off', 0))}"
    )
    ax_comp.text(
        0.02,
        0.98,
        summary,
        transform=ax_comp.transAxes,
        va="top",
        ha="left",
        fontsize=8.7,
        bbox={"facecolor": "white", "alpha": 0.7, "linewidth": 0},
    )

    fig.suptitle(
        "Tamil Nadu / Puducherry AIS PoC: ban-compliance index + hotspot persistence",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DASHBOARD, dpi=220, bbox_inches="tight")
    print(f"Saved -> {OUT_DASHBOARD}")


def export_interactive_map(
    classified: pd.DataFrame,
    boundary: gpd.GeoSeries,
    zone: gpd.GeoSeries,
    persistence: pd.DataFrame,
) -> None:
    center = [(BBOX[1] + BBOX[3]) / 2, (BBOX[0] + BBOX[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")

    folium.GeoJson(
        zone.to_json(),
        name=f"{ARTISANAL_NM} nm artisanal marine zone",
        style_function=lambda _: {
            "color": "cyan",
            "weight": 1.5,
            "fill": True,
            "fillOpacity": 0.18,
        },
    ).add_to(fmap)
    folium.GeoJson(
        boundary.to_json(),
        name=f"{ARTISANAL_NM} nm boundary",
        style_function=lambda _: {"color": "cyan", "weight": 2.0, "fill": False},
    ).add_to(fmap)

    inside = classified[classified["inside_artisanal_zone"]]
    outside = classified[~classified["inside_artisanal_zone"]]
    if not inside.empty:
        HeatMap(
            inside[["lat", "lon", "hours"]].to_numpy().tolist(),
            name="Heatmap: inside 3 nm zone",
            radius=11,
            blur=18,
            min_opacity=0.25,
            max_zoom=10,
            gradient={0.2: "#ffe5d9", 0.5: "#f04a3a", 1.0: "#7a0019"},
        ).add_to(fmap)
    if not outside.empty:
        HeatMap(
            outside[["lat", "lon", "hours"]].to_numpy().tolist(),
            name="Heatmap: outside 3 nm zone",
            radius=9,
            blur=16,
            min_opacity=0.2,
            max_zoom=10,
            gradient={0.2: "#edf8fb", 0.5: "#43a2ca", 1.0: "#0868ac"},
        ).add_to(fmap)

    for _, row in persistence.head(30).iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(3, int(row["active_months"])),
            color="black",
            weight=1,
            fill=True,
            fill_opacity=0.15,
            tooltip=(
                f"{row['persistence_class']} | months={int(row['active_months'])} | "
                f"hours={row['total_hours']:.1f}"
            ),
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]])
    fmap.save(str(OUT_HTML))
    print(f"Saved -> {OUT_HTML}")


def main() -> None:
    monthly_cells = fetch_monthly_effort()
    boundary, zone = build_artisanal_zone()
    classified = classify_in_zone(monthly_cells, zone)
    monthly = build_monthly_summary(classified)
    compliance = build_ban_compliance_summary(monthly)
    persistence = build_hotspot_persistence(classified)
    save_tables(monthly, compliance, persistence)
    plot_dashboard(classified, boundary, zone, monthly, compliance, persistence)
    export_interactive_map(classified, boundary, zone, persistence)


if __name__ == "__main__":
    main()
