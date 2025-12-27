import cupy as cp
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import contextily as ctx
import pyproj


# GPU-обёртка для проекции координат
def project_coordinates(x_vals, y_vals):
    """Проекция координат из EPSG:27700 в EPSG:4326"""
    x_cpu, y_cpu = cp.asnumpy(x_vals), cp.asnumpy(y_vals)
    transformer = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon_cpu, lat_cpu = transformer.transform(x_cpu, y_cpu)
    return cp.asarray(lat_cpu), cp.asarray(lon_cpu)

coord_x = cp.asarray(gdf['easting'].to_numpy())
coord_y = cp.asarray(gdf['northing'].to_numpy())
lat, lon = project_coordinates(coord_x, coord_y)

gdf['lat_wgs84'], gdf['lon_wgs84'] = lat.get(), lon.get()
cluster_total = int(gdf['cluster'].nunique())
print(f"Количество кластеров в данных: {cluster_total}")
cluster_index = gdf['cluster'].unique().to_arrow().to_pylist()


def build_cluster_visualization(frame, img_resolution=1000):
    cluster_num = int(frame['cluster'].nunique())
    work_df = frame.copy(deep=True)
    work_df['cluster'] = work_df['cluster'].astype('category')
    bounds_x = (float(work_df['lon_wgs84'].min()), float(work_df['lon_wgs84'].max()))
    bounds_y = (float(work_df['lat_wgs84'].min()), float(work_df['lat_wgs84'].max()))
    raster_space = ds.Canvas(plot_width=img_resolution, plot_height=img_resolution,
                                x_range=bounds_x, y_range=bounds_y)
    density_map = raster_space.points(work_df, x='lon_wgs84', y='lat_wgs84', agg=ds.count_cat('cluster'))
    base_colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'brown',
        'pink', 'gray', 'olive', 'cyan', 'magenta', 'navy']
    color_scheme = base_colors[:cluster_num]
    colored_raster = tf.shade(density_map, color_key=color_scheme,
                              how='eq_hist', min_alpha=150)
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(bounds_x)
    ax.set_ylim(bounds_y)

    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    ax.imshow(colored_raster.to_pil(),
              extent=[bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]],
              interpolation='nearest', aspect='auto', alpha=0.8)

    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    ax.set_title('Пространственная кластеризация точек (UK)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.4)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', markerfacecolor=clr,
                   markeredgecolor='w', markersize=10, label=f'Кластер {idx}')
        for idx, clr in enumerate(color_scheme)
    ]
    ax.legend(handles=legend_handles, loc='upper right', title='Кластеры')
    plt.tight_layout()
    return fig


geo_lat_limits = (49.5, 61.0)
geo_lon_limits = (-8.5, 2.0)
final_plot = build_cluster_visualization(gdf)
plt.show()
