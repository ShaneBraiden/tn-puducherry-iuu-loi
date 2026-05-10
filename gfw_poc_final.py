"""
Enhanced GFW AIS proof-of-concept for Tamil Nadu / Puducherry.

What this adds beyond gfw_demo.py:
1. Monthly retrieval (sample temporal analysis, not only aggregate heatmap).
2. 3 nm artisanal-zone classification (inside vs outside legal zone).
3. Multi-panel PNG "dashboard" + interactive HTML + CSV summary tables.

Run:
  1. pip install -r requirements.txt
  2. Set GFW_TOKEN in shell
  3. python gfw_poc_sample.py
"""

from __future__ import annotations

import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import folium
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import requests
from folium.plugins import HeatMap
from shapely.geometry import box, mapping

# Non-GUI backend (safe for headless environments)
mpl.use("Agg")


# ---- CONFIG ---------------------------------------------------------------

GFW_TOKEN = os.environ.get("GFW_TOKEN", "").strip()

# WGS 84 / EPSG:4326
# Southwest: 11.800683, 79.549805
# Northeast: 13.285927, 80.566040
BBOX = (79.549805, 11.800683, 80.566040, 13.285927)  # (W, S, E, N)

# Sample window for PoC (monthly slices)
START_MONTH = "2023-01"
END_MONTH = "2024-12"
BAN_MONTHS = {4, 5, 6}  # East coast mechanised fishing ban (Apr 15 - Jun 14)

DATASET = "public-global-fishing-effort:latest"
SPATIAL_RES = "HIGH"
ARTISANAL_NM = 5
NM_TO_M = 1852

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DASHBOARD = OUT_DIR / "poc_ais_dashboard_tn_puducherry.png"
OUT_HTML = OUT_DIR / "poc_ais_dashboard_tn_puducherry.html"
OUT_MONTHLY_CSV = OUT_DIR / "poc_monthly_summary_tn_puducherry.csv"
OUT_HOTSPOTS_CSV = OUT_DIR / "poc_hotspots_in_zone_tn_puducherry.csv"
GFW_REPORT_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
LOG_API_RESPONSE_PREVIEW = True


# ---- API HELPERS ----------------------------------------------------------

def _month_ranges(start_month: str, end_month: str) -> list[tuple[str, str, str]]:
    periods = pd.period_range(start=start_month, end=end_month, freq="M")
    ranges = []
    for p in periods:
        start = p.start_time.strftime("%Y-%m-%d")
        end = p.end_time.strftime("%Y-%m-%d")
        ranges.append((start, end, str(p)))
    return ranges


def _normalise_cell_key(name: str) -> str:
    key = name.lower()
    key_map = {
        "lat": "lat",
        "latitude": "lat",
        "cell_ll_lat": "lat",
        "lon": "lon",
        "lng": "lon",
        "longitude": "lon",
        "cell_ll_lon": "lon",
        "hours": "hours",
        "fishing_hours": "hours",
        "apparent_fishing_hours": "hours",
        "value": "hours",
    }
    return key_map.get(key, key)


def _extract_cell_records(node: object, sink: list[dict]) -> None:
    if isinstance(node, dict):
        lowered_keys = {_normalise_cell_key(str(k)) for k in node.keys()}
        if {"lat", "lon", "hours"}.issubset(lowered_keys):
            sink.append(node)
            return
        for value in node.values():
            _extract_cell_records(value, sink)
        return
    if isinstance(node, list):
        for item in node:
            _extract_cell_records(item, sink)


def _log_payload_preview(period_label: str, payload: object) -> None:
    if not LOG_API_RESPONSE_PREVIEW:
        return
    if isinstance(payload, dict):
        top_keys = list(payload.keys())
        entries = payload.get("entries")
        entries_len = len(entries) if isinstance(entries, list) else None
        first_entry = entries[0] if isinstance(entries, list) and entries else None
        first_entry_keys = list(first_entry.keys()) if isinstance(first_entry, dict) else None
        print(
            f"    response[{period_label}] keys={top_keys} "
            f"entries={entries_len} first_entry_keys={first_entry_keys}"
        )
        return
    print(f"    response[{period_label}] type={type(payload).__name__}")


def _is_explicit_no_data_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return False
    has_dataset_wrapper = False
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        has_dataset_wrapper = True
        if not all(v is None for v in entry.values()):
            return False
    return has_dataset_wrapper


def _parse_4wings_cells(payload: object) -> pd.DataFrame:
    if _is_explicit_no_data_payload(payload):
        return pd.DataFrame(columns=["lat", "lon", "hours"])

    records: list[dict] = []
    _extract_cell_records(payload, records)
    if not records:
        raise SystemExit(
            f"Unexpected GFW response shape. First 400 chars:\n{str(payload)[:400]}"
        )

    normalised = [{_normalise_cell_key(str(k)): v for k, v in rec.items()} for rec in records]
    df = pd.DataFrame(normalised)
    if not {"lat", "lon", "hours"}.issubset(df.columns):
        raise SystemExit(
            f"Missing lat/lon/hours columns after normalising. Got: {list(df.columns)}"
        )

    out = df[["lat", "lon", "hours"]].copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["hours"] = pd.to_numeric(out["hours"], errors="coerce")
    out = out.dropna(subset=["lat", "lon", "hours"])
    if out.empty:
        raise SystemExit("Parsed 4Wings response but no valid numeric lat/lon/hours rows.")
    return out


def fetch_one_month(start: str, end: str, period_label: str) -> pd.DataFrame:
    params = {
        "format": "JSON",
        "temporal-resolution": "ENTIRE",
        "spatial-resolution": SPATIAL_RES,
        "datasets[0]": DATASET,
        "date-range": f"{start},{end}",
        "spatial-aggregation": "false",
    }
    body = {"geojson": mapping(box(*BBOX))}
    headers = {
        "Authorization": f"Bearer {GFW_TOKEN}",
        "Content-Type": "application/json",
    }
    print(f"  -> {period_label} ({start} to {end})", flush=True)
    resp = requests.post(
        GFW_REPORT_URL, params=params, json=body, headers=headers, timeout=600
    )
    if resp.status_code != 200:
        raise SystemExit(f"GFW API error {resp.status_code}: {resp.text[:600]}")

    payload = resp.json()
    _log_payload_preview(period_label, payload)
    out = _parse_4wings_cells(payload)
    if out.empty:
        print(f"    response[{period_label}] has no AIS cells for this month")
    out["period"] = period_label
    return out


def fetch_monthly_effort() -> pd.DataFrame:
    if not GFW_TOKEN:
        raise SystemExit(
            "GFW_TOKEN env var is empty.\n"
            "Get a free token at https://globalfishingwatch.org/our-apis/tokens "
            "and set it in your shell."
        )

    print("Fetching monthly AIS apparent fishing effort:")
    parts = [
        fetch_one_month(start, end, period)
        for start, end, period in _month_ranges(START_MONTH, END_MONTH)
    ]
    non_empty_parts = [p for p in parts if not p.empty]
    if not non_empty_parts:
        raise SystemExit("No AIS effort cells were returned for the selected months.")

    df = pd.concat(non_empty_parts, ignore_index=True)
    df = df.groupby(["period", "lat", "lon"], as_index=False)["hours"].sum()
    print(
        f"  -> {len(df):,} monthly grid-cells, "
        f"{df['hours'].sum():,.0f} fishing-hours total"
    )
    return df


# ---- SPATIAL CLASSIFICATION ----------------------------------------------

def build_artisanal_zone() -> tuple[gpd.GeoSeries, gpd.GeoSeries]:
    """
    Return:
      1) artisanal outer boundary (3 nm from coast)
      2) artisanal marine zone polygon (sea area within 3 nm from coast)
    """
    shp = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    land = gpd.read_file(shp)

    pad = 1.0
    study = box(BBOX[0] - pad, BBOX[1] - pad, BBOX[2] + pad, BBOX[3] + pad)
    land_local = gpd.clip(land, study)
    if land_local.empty:
        raise SystemExit("No land geometry found near study area.")

    land_utm = land_local.to_crs(32644)
    land_geom = land_utm.unary_union
    buffered = land_geom.buffer(ARTISANAL_NM * NM_TO_M)
    zone_geom = buffered.difference(land_geom)

    boundary = gpd.GeoSeries([buffered.boundary], crs=32644).to_crs(4326)
    zone = gpd.GeoSeries([zone_geom], crs=32644).to_crs(4326)
    return boundary, zone


def classify_in_zone(df: pd.DataFrame, zone: gpd.GeoSeries) -> pd.DataFrame:
    zone_union = zone.unary_union
    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=4326,
    )
    pts["inside_artisanal_zone"] = pts.geometry.within(zone_union)
    pts["period_dt"] = pd.to_datetime(pts["period"] + "-01")
    return pd.DataFrame(pts.drop(columns="geometry"))


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["period", "inside_artisanal_zone"], as_index=False)["hours"]
        .sum()
        .pivot(index="period", columns="inside_artisanal_zone", values="hours")
        .fillna(0.0)
        .rename(columns={False: "outside_hours", True: "inside_hours"})
        .reset_index()
    )
    if "inside_hours" not in monthly.columns:
        monthly["inside_hours"] = 0.0
    if "outside_hours" not in monthly.columns:
        monthly["outside_hours"] = 0.0
    monthly["total_hours"] = monthly["inside_hours"] + monthly["outside_hours"]
    monthly["inside_share_pct"] = (
        100.0 * monthly["inside_hours"] / monthly["total_hours"].replace(0, pd.NA)
    ).fillna(0.0)
    monthly["period_dt"] = pd.to_datetime(monthly["period"] + "-01")
    monthly["is_ban_month"] = monthly["period_dt"].dt.month.isin(BAN_MONTHS)
    return monthly.sort_values("period").reset_index(drop=True)


# ---- OUTPUTS --------------------------------------------------------------

def save_tables(classified: pd.DataFrame, monthly: pd.DataFrame) -> None:
    export_monthly = monthly[
        [
            "period",
            "inside_hours",
            "outside_hours",
            "total_hours",
            "inside_share_pct",
            "is_ban_month",
        ]
    ].copy()
    export_monthly.to_csv(OUT_MONTHLY_CSV, index=False)

    hotspots = (
        classified[classified["inside_artisanal_zone"]]
        .groupby(["lat", "lon"], as_index=False)["hours"]
        .sum()
        .sort_values("hours", ascending=False)
        .head(25)
    )
    hotspots.to_csv(OUT_HOTSPOTS_CSV, index=False)

    print(f"Saved -> {OUT_MONTHLY_CSV}")
    print(f"Saved -> {OUT_HOTSPOTS_CSV}")


def plot_dashboard(
    classified: pd.DataFrame,
    boundary: gpd.GeoSeries,
    zone: gpd.GeoSeries,
    monthly: pd.DataFrame,
) -> None:
    classified = classified[classified["hours"] > 0].copy()
    if classified.empty:
        raise SystemExit("Response had no non-zero fishing-effort cells.")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0])
    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax_month = fig.add_subplot(gs[0, 1])
    ax_ban = fig.add_subplot(gs[1, 1])

    ax_map.set_extent((BBOX[0], BBOX[2], BBOX[1], BBOX[3]), crs=ccrs.PlateCarree())
    ax_map.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "10m"),
        facecolor="#f5f9ff",
        zorder=1,
    )
    # 3. Add land as greenish-brown so it is clearly distinct from water
    ax_map.add_feature(
        cfeature.LAND,
        facecolor="#a89060",
        edgecolor="#444444",
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
        vmin=max(classified["hours"].quantile(0.05), 0.1),
        vmax=classified["hours"].quantile(0.99),
    )
    sc = ax_map.scatter(
        classified["lon"],
        classified["lat"],
        c=classified["hours"],
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

    ax_map.add_feature(
        cfeature.NaturalEarthFeature("physical", "coastline", "10m"),
        linewidth=0.6,
        zorder=4,
    )

    # Place markers and labels for key coastal reference cities
    cities = {
        "Chennai": (80.2707, 13.0827),
        "Mahabalipuram": (80.1927, 12.6208),
        "Puducherry": (79.8145, 11.9139),
    }
    for name, (clon, clat) in cities.items():
        ax_map.plot(
            clon,
            clat,
            marker="o",
            color="black",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            transform=ccrs.PlateCarree(),
            zorder=9,
        )
        ax_map.text(
            clon - 0.04,
            clat + 0.04,
            name,
            fontsize=8,
            fontweight="bold",
            color="black",
            ha="right",
            va="bottom",
            transform=ccrs.PlateCarree(),
            zorder=10,
            bbox=dict(facecolor="white", alpha=0.7, linewidth=0, pad=1.2),
        )

    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax_map.set_title("AIS apparent fishing effort cells + 5 nm legal-zone overlay")

    cbar = plt.colorbar(sc, ax=ax_map, shrink=0.65, pad=0.03)
    cbar.set_label("Fishing hours (log scale)")

    ax_month.plot(
        monthly["period_dt"],
        monthly["total_hours"],
        label="Total",
        color="#2b8cbe",
        linewidth=1.8,
    )
    ax_month.plot(
        monthly["period_dt"],
        monthly["inside_hours"],
        label="Inside 3 nm zone",
        color="#d7301f",
        linewidth=1.5,
    )
    
    # Exact dates for East Coast continuous ban: April 15 to June 14
    years_in_data = monthly["period_dt"].dt.year.unique()
    for y in years_in_data:
        ban_start = pd.Timestamp(year=y, month=4, day=15)
        ban_end = pd.Timestamp(year=y, month=6, day=14)
        ax_month.axvspan(ban_start, ban_end, color="#fee391", alpha=0.25, linewidth=0)
        
    ax_month.set_title("Monthly effort trend (yellow = Apr 15 - Jun 14 ban)")
    ax_month.set_ylabel("Fishing hours")
    ax_month.grid(alpha=0.25, linestyle="--")
    ax_month.legend(frameon=False)

    tmp = monthly.copy()
    tmp["year"] = tmp["period_dt"].dt.year
    by_year = (
        tmp.groupby(["year", "is_ban_month"], as_index=False)[
            ["inside_hours", "total_hours"]
        ]
        .sum()
        .sort_values(["year", "is_ban_month"])
    )
    by_year["inside_share_pct"] = (
        100.0 * by_year["inside_hours"] / by_year["total_hours"].replace(0, pd.NA)
    ).fillna(0.0)

    years = sorted(by_year["year"].unique().tolist())
    x = range(len(years))
    ban_vals = [
        float(
            by_year[
                (by_year["year"] == y) & (by_year["is_ban_month"] == True)
            ]["inside_share_pct"].sum()
        )
        for y in years
    ]
    non_ban_vals = [
        float(
            by_year[
                (by_year["year"] == y) & (by_year["is_ban_month"] == False)
            ]["inside_share_pct"].sum()
        )
        for y in years
    ]

    w = 0.38
    ax_ban.bar([i - w / 2 for i in x], non_ban_vals, width=w, label="Non-ban months")
    ax_ban.bar([i + w / 2 for i in x], ban_vals, width=w, label="Ban months")
    ax_ban.set_xticks(list(x), [str(y) for y in years])
    ax_ban.set_ylim(bottom=0)
    ax_ban.set_ylabel("Inside-zone share (%)")
    ax_ban.set_title("Sample compliance signal: inside-zone share by season")
    ax_ban.grid(alpha=0.25, linestyle="--", axis="y")
    ax_ban.legend(frameon=False)

    fig.suptitle(
        "Tamil Nadu / Puducherry AIS PoC: spatial hotspots + temporal compliance sample",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DASHBOARD, dpi=220, bbox_inches="tight")
    print(f"Saved -> {OUT_DASHBOARD}")


def export_interactive_map(
    classified: pd.DataFrame, boundary: gpd.GeoSeries, zone: gpd.GeoSeries
) -> None:
    center = [(BBOX[1] + BBOX[3]) / 2, (BBOX[0] + BBOX[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")

    zone_layer = folium.FeatureGroup(name=f"{ARTISANAL_NM} nm artisanal marine zone")
    folium.GeoJson(
        zone.to_json(),
        style_function=lambda _: {
            "color": "cyan",
            "weight": 1.5,
            "fill": True,
            "fillOpacity": 0.18,
        },
    ).add_to(zone_layer)
    zone_layer.add_to(fmap)

    boundary_layer = folium.FeatureGroup(name=f"{ARTISANAL_NM} nm boundary")
    folium.GeoJson(
        boundary.to_json(),
        style_function=lambda _: {"color": "cyan", "weight": 2.0, "fill": False},
    ).add_to(boundary_layer)
    boundary_layer.add_to(fmap)

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

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]])
    fmap.save(str(OUT_HTML))
    print(f"Saved -> {OUT_HTML}")


def main() -> None:
    monthly_cells = fetch_monthly_effort()
    boundary, zone = build_artisanal_zone()
    classified = classify_in_zone(monthly_cells, zone)
    monthly = build_monthly_summary(classified)
    save_tables(classified, monthly)
    plot_dashboard(classified, boundary, zone, monthly)
    export_interactive_map(classified, boundary, zone)


if __name__ == "__main__":
    main()
