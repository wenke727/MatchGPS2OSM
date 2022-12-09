import geopandas as gpd
from sqlalchemy import create_engine

IP = "192.168.135.16"
connect= f"postgresql://postgres:pcl_A5A@{IP}:5432/gis"
ENGINE   = create_engine(connect)


def gdf_to_postgis(gdf, name, engine=ENGINE, if_exists='replace', *args, **kwargs):
    """Save the GeoDataFrame to the db

    Args:
        gdf ([type]): [description]
        name ([type]): [description]
        engine ([type], optional): [description]. Defaults to ENGINE.
        if_exists (str, optional): [description]. Defaults to 'replace'. if_exists{‘fail’, ‘replace’, ‘append’}

    Returns:
        [type]: [description]
    """
    gdf.to_postgis(name=name, con=engine, if_exists=if_exists)


def gdf_to_geojson(gdf, fn):
    if not isinstance(gdf, gpd.GeoDataFrame):
        print('Check the format of the gdf.')
        return False

    if 'geojson' not in fn:
        fn = f'{fn}.geojson'
    
    gdf.to_file(fn, driver="GeoJSON")

    return 

