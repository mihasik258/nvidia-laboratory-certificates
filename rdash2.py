import cupy as cp
import pyproj

# Конвертация координат
gdf = infected_df.copy()
cupy_east = cp.asarray(gdf['easting'])
cupy_north = cp.asarray(gdf['northing'])

def utm_to_lat_lon(easting, northing):
    easting_np = cp.asnumpy(easting)
    northing_np = cp.asnumpy(northing)
    transformer = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True) 
    lon_np, lat_np = transformer.transform(easting_np, northing_np)
    return cp.asarray(lat_np), cp.asarray(lon_np)

gdf['lat'], gdf['long'] = utm_to_lat_lon(cupy_east, cupy_north)

new_palette = [
    "#FF5E5B", "#00CECB", "#FFED66", "#6C5B7B", "#355C7D",
    "#F67280", "#C06C84", "#6C5B7B", "#99B898", "#FECEAB"
]

infected_palette = [
    "#F8F3D4", "#F6416C", "#FF9A00", "#00B8A9", "#2D4059"
]

cxf_data = cxf.DataFrame.from_dataframe(gdf)

scatter_chart1 = cxf.charts.scatter(
    x='long', 
    y='lat', 
    tile_provider="StamenTonerBackground",
    aggregate_col='cluster', 
    pixel_shade_type='density',
    point_size=8,
    aggregate_fn='median',
    color_palette=new_palette,
    stroke_width=1.5,
    stroke_color="#FFFFFF",
    point_shape='triangle',
    blending_mode='multiply',
)

scatter_chart2 = cxf.charts.scatter(
    x='long', 
    y='lat', 
    tile_provider="CartoDBPositron",
    aggregate_col='infected', 
    pixel_shade_type='exponential',
    point_size=7,
    aggregate_fn='count',
    color_palette=infected_palette,
    stroke_width=0.8,
    stroke_color="#333333",
    point_shape='triangle',
    blending_mode='multiply',
)

cluster_selector = cxf.charts.panel_widgets.range_slider(
    'cluster',
    min_value=int(gdf['cluster'].min()),
    max_value=int(gdf['cluster'].max()),
    step=1,
    value=[int(gdf['cluster'].min()), int(gdf['cluster'].max())],
    orientation='vertical',
    height=300
)

display_mode = cxf.charts.panel_widgets.radio_group(
    'display_mode',
    options=['Normal', 'Heatmap Overlay', 'Contour Lines'],
    value='Normal'
)

layout_array = [
    [1],
    [2]
]

dash = cxf_data.dashboard(
    charts=[scatter_chart1, scatter_chart2],
    sidebar=[cluster_selector, display_mode],
    layout_array=layout_array,
    theme="minimal",
)

dash.app(
    port=8050,
    debug=True,
    mode='inline' if hasattr(cxf, '__file__') else 'external'
)
