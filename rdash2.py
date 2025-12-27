import cupy as cp
import pyproj
import contextily as ctx
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

gdf = infected_df.copy(deep=True)
coord_x = cp.asarray(gdf["easting"].values)
coord_y = cp.asarray(gdf["northing"].values)

def project_coordinates(x_vals, y_vals):
    x_cpu = cp.asnumpy(x_vals)
    y_cpu = cp.asnumpy(y_vals)
    transformer = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:3857", always_xy=True)
    lon_cpu, lat_cpu = transformer.transform(x_cpu, y_cpu)
    return cp.asarray(lat_cpu), cp.asarray(lon_cpu)

lat, lon = project_coordinates(coord_x, coord_y)
gdf["lat_wgs84"] = lat
gdf["lon_wgs84"] = lon

color_set = ["#3366cc", "#66aa00", "#dc3912", "#ff9900", "#109618",
             "#0099c6", "#ddad4f", "#6a4c93", "#ff6e54", "#a4c2f4"]

cx_data = cxf.DataFrame.from_dataframe(gdf)

map_by_cluster = cxf.charts.scatter(x="lon_wgs84", y="lat_wgs84", aggregate_col="cluster",
                                    aggregate_fn="mean", color_palette=color_set, point_size=5,
                                    tile_provider="OSM", pixel_shade_type="linear", unselected_alpha=0)

map_by_density = cxf.charts.scatter(x="lon_wgs84", y="lat_wgs84", aggregate_col="infected",
                                    aggregate_fn="count", point_size=5, tile_provider="OSM",
                                    pixel_shade_type="linear", unselected_alpha=0)

cluster_filter = cxf.charts.panel_widgets.multi_select(x="cluster")

dashboard = cx_data.dashboard(charts=[map_by_cluster, map_by_density],
                              sidebar=[cluster_filter], layout_array=[[1, 2]])

dashboard.app()
