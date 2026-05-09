"""
GFW preliminary fishing-effort retrieval - Tamil Nadu / Puducherry coast.

Pulls AIS apparent fishing effort from the Global Fishing Watch 4Wings API
for the LOI study bounding box (SW: 11.800683, 79.549805; NE: 13.285927, 80.566040), aggregates
2023-2024, and saves a heatmap PNG with the 3 nm artisanal-zone boundary
(Tamil Nadu Marine Fisheries Regulation Act 1983) overlaid.

Setup:
  1. Get a free GFW API token: https://globalfishingwatch.org/our-apis/tokens
  2. pip install -r requirements.txt
  3. Set the token in your shell:
       PowerShell : $env:GFW_TOKEN = "<your-token>"
       cmd.exe    : set GFW_TOKEN=<your-token>
       bash       : export GFW_TOKEN=<your-token>
  4. python gfw_demo.py

Output: figure_fishing_effort_tn_puducherry.png
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
# Use a non-GUI backend to avoid Qt platform plugin issues in headless
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.plugins import HeatMap
from shapely.geometry import box, mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# ---- CONFIG -------------------------------------------------------------

GFW_TOKEN = os.environ.get("GFW_TOKEN", "").strip()

# Bbox (WGS 84 / EPSG:4326): SW(11.800683, 79.549805), NE(13.285927, 80.566040).
BBOX = (79.549805, 11.800683, 80.566040, 13.285927)  # (W, S, E, N)

# Two-year window, split because the 4Wings date-range param caps at 366 days.
DATE_RANGES = [
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

DATASET = "public-global-fishing-effort:latest"
SPATIAL_RES = "HIGH"                        # 0.01 deg ~= 1 km at this latitude
ARTISANAL_NM = 3
NM_TO_M = 1852

OUT_FIG = Path(__file__).parent / "figure_fishing_effort_tn_puducherry.png"
OUT_HTML = Path(__file__).parent / "figure_fishing_effort_tn_puducherry.html"
GFW_REPORT_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"


# ---- DATA PULL ----------------------------------------------------------

def _parse_4wings_response(payload) -> pd.DataFrame:
    """Tolerate the two known shapes the 4Wings JSON response can take."""
    cells: list = []
    if isinstance(payload, list):
        cells = payload
    elif isinstance(payload, dict):
        entries = payload.get("entries")
        if entries:
            for entry in entries:
                if isinstance(entry, dict):
                    for v in entry.values():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            cells.extend(v)
                            break
        else:
            for v in payload.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    cells = v
                    break

    if not cells:
        raise SystemExit(
            f"Unexpected GFW response shape. First 400 chars:\n{str(payload)[:400]}"
        )

    df = pd.DataFrame(cells)
    df.columns = [c.lower() for c in df.columns]
    if not {"lat", "lon", "hours"}.issubset(df.columns):
        raise SystemExit(f"Missing lat/lon/hours columns. Got: {list(df.columns)}")
    return df[["lat", "lon", "hours"]]


def fetch_one(start: str, end: str) -> pd.DataFrame:
    region_geojson = mapping(box(*BBOX))
    params = {
        "format": "JSON",
        "temporal-resolution": "ENTIRE",
        "spatial-resolution": SPATIAL_RES,
        "datasets[0]": DATASET,
        "date-range": f"{start},{end}",
        "spatial-aggregation": "false",
    }
    body = {"geojson": region_geojson}
    headers = {
        "Authorization": f"Bearer {GFW_TOKEN}",
        "Content-Type": "application/json",
    }
    print(f"  -> {start} ... {end}", flush=True)
    r = requests.post(
        GFW_REPORT_URL, params=params, json=body, headers=headers, timeout=600
    )
    if r.status_code != 200:
        raise SystemExit(f"GFW API error {r.status_code}: {r.text[:600]}")
    return _parse_4wings_response(r.json())


def fetch_all() -> pd.DataFrame:
    if not GFW_TOKEN:
        raise SystemExit(
            "GFW_TOKEN env var is empty.\n"
            "Get a free token at https://globalfishingwatch.org/our-apis/tokens "
            "and set it in your shell."
        )
    print("Fetching GFW AIS apparent fishing effort:")
    parts = [fetch_one(s, e) for s, e in DATE_RANGES]
    df = pd.concat(parts, ignore_index=True)
    df = df.groupby(["lat", "lon"], as_index=False)["hours"].sum()
    print(f"  -> {len(df):,} cells, {df['hours'].sum():,.0f} fishing-hours total")
    return df


# ---- 3 NM ARTISANAL ZONE -----------------------------------------------

def build_artisanal_boundary() -> gpd.GeoSeries:
    """Outer edge of the 3 nm artisanal zone, in WGS84."""
    shp = shpreader.natural_earth(
        resolution="10m", category="physical", name="land"
    )
    land = gpd.read_file(shp)
    pad = 1.0
    study = box(BBOX[0] - pad, BBOX[1] - pad, BBOX[2] + pad, BBOX[3] + pad)
    land_local = gpd.clip(land, study)
    # UTM 44N (EPSG:32644) gives a metric CRS over the TN / Puducherry coast,
    # so a constant-distance buffer in metres is geodesically faithful.
    land_utm = land_local.to_crs(32644)
    buffered = land_utm.buffer(ARTISANAL_NM * NM_TO_M)
    return gpd.GeoSeries(buffered.boundary, crs=32644).to_crs(4326)


# ---- PLOT ---------------------------------------------------------------

def plot(df: pd.DataFrame, artisanal: gpd.GeoSeries) -> None:
    df = df[df["hours"] > 0].copy()
    if df.empty:
        raise SystemExit("Response had no non-zero fishing-effort cells.")

    grid = df.pivot_table(
        index="lat", columns="lon", values="hours", aggfunc="sum"
    )
    lons, lats, data = grid.columns.values, grid.index.values, grid.values

    fig = plt.figure(figsize=(8, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # `BBOX` is defined as (west, south, east, north).
    # Cartopy `set_extent` expects (west, east, south, north) when given as
    # (x0, x1, y0, y1). Reorder accordingly to avoid showing the wrong region.
    ax.set_extent((BBOX[0], BBOX[2], BBOX[1], BBOX[3]), crs=ccrs.PlateCarree())

    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "10m"),
        facecolor="#f5f9ff", zorder=1,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "10m",
            edgecolor="black", facecolor="#dcdcdc",
        ),
        zorder=2,
    )

    norm = mcolors.LogNorm(
        vmin=max(df["hours"].quantile(0.05), 0.1),
        vmax=df["hours"].quantile(0.99),
    )

    # For clarity and to ensure each data cell is plotted at the correct
    # geographic location, plot the aggregated cells as a scatter of lon/lat
    # centres rather than using pcolormesh. This also makes it easier to
    # verify that points lie on water or land.
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=4326,
    )

    # Load Natural Earth land clipped to the study area to test point locations
    shp = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    land = gpd.read_file(shp)
    study = box(BBOX[0] - 1.0, BBOX[1] - 1.0, BBOX[2] + 1.0, BBOX[3] + 1.0)
    land_local = gpd.clip(land, study).to_crs(4326)

    # Spatial join: mark points that fall on land
    try:
        pts_on_land = pts.sjoin(land_local, how="left", predicate="within")
        on_land_mask = pts_on_land.index_right.notna()
    except Exception:
        # Fallback: if spatial join isn't available in the environment,
        # conservatively assume none are on land.
        on_land_mask = [False] * len(pts)

    n_on_land = int(sum(on_land_mask))
    n_total = len(pts)
    print(f"  -> {n_on_land} of {n_total} aggregated cells fall on land (based on Natural Earth clip)")

    sc = ax.scatter(
        pts["lon"], pts["lat"], c=pts["hours"],
        cmap="hot_r", norm=norm, s=14, alpha=0.85,
        transform=ccrs.PlateCarree(), zorder=6, linewidths=0,
    )

    for geom in artisanal:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for part in parts:
            x, y = part.xy
            ax.plot(
                x, y, color="cyan", linewidth=1.2,
                transform=ccrs.PlateCarree(), zorder=5,
            )

    # Add city markers (lon, lat) and labels
    cities = {
        "Chennai": (80.2707, 13.0827),
        "Puducherry": (79.8145, 11.9139),
    }
    for name, (clon, clat) in cities.items():
        ax.plot(clon, clat, marker="o", color="black", markersize=5,
                transform=ccrs.PlateCarree(), zorder=8)
        ax.text(clon + 0.06, clat + 0.06, name, fontsize=8,
                transform=ccrs.PlateCarree(), zorder=8,
                bbox=dict(facecolor="white", alpha=0.6, linewidth=0, pad=1))

    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "coastline", "10m"),
        linewidth=0.6, zorder=4,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.04)
    cbar.set_label("Apparent fishing hours, 2023-2024 (log scale)")

    ax.set_title(
        "GFW AIS apparent fishing effort - Tamil Nadu & Puducherry coast\n"
        f"{DATE_RANGES[0][0]} to {DATE_RANGES[-1][1]}  -  "
        f"cyan = {ARTISANAL_NM} nm artisanal-zone boundary (TNMFR Act 1983)",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"Saved -> {OUT_FIG}")


def export_interactive_map(df: pd.DataFrame, artisanal: gpd.GeoSeries) -> None:
    """Export an interactive Leaflet map with heat spots and boundary overlays."""
    df = df[df["hours"] > 0].copy()
    if df.empty:
        raise SystemExit("Response had no non-zero fishing-effort cells.")

    center_lat = (BBOX[1] + BBOX[3]) / 2
    center_lon = (BBOX[0] + BBOX[2]) / 2
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="CartoDB positron",
        control_scale=True,
    )

    heat_data = df[["lat", "lon", "hours"]].to_numpy().tolist()
    HeatMap(
        heat_data,
        name="Fishing effort heat",
        radius=10,
        blur=18,
        max_zoom=9,
        min_opacity=0.25,
        max_val=float(df["hours"].quantile(0.99)),
        gradient={0.2: "#ffffcc", 0.4: "#ffeda0", 0.6: "#feb24c", 0.8: "#f03b20", 1.0: "#800026"},
    ).add_to(fmap)

    folium.GeoJson(
        artisanal.to_json(),
        name=f"{ARTISANAL_NM} nm artisanal boundary",
        style_function=lambda _: {
            "color": "cyan",
            "weight": 2,
            "fill": False,
        },
    ).add_to(fmap)

    cities = {
        "Chennai": (80.2707, 13.0827),
        "Puducherry": (79.8145, 11.9139),
    }
    for name, (clon, clat) in cities.items():
        folium.Marker(
            location=[clat, clon],
            tooltip=name,
            popup=name,
            icon=folium.Icon(color="black", icon="info-sign"),
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]])
    fmap.save(str(OUT_HTML))
    print(f"Saved -> {OUT_HTML}")


def main() -> None:
    df = fetch_all()
    artisanal = build_artisanal_boundary()
    plot(df, artisanal)
    export_interactive_map(df, artisanal)


if __name__ == "__main__":
    main()
