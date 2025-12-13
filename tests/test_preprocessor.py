import pandas as pd
import numpy as np
from src.preprocessor import DataCleaner, GeoDistanceTransformer

def test_geo_distance_transformer(sample_df):
    center = (55.7558, 37.6173)
    transformer = GeoDistanceTransformer(*center)
    
    res_df = transformer.transform(sample_df)
    
    assert 'distance_to_center' in res_df.columns
    assert not res_df['distance_to_center'].isnull().all()
    assert res_df.loc[0, 'distance_to_center'] < 10.0

def test_data_cleaner_fill(sample_df):
    cleaner = DataCleaner()
    cleaner.fit(sample_df)
    res_df = cleaner.transform(sample_df)
    
    assert not res_df['rooms'].isnull().any()
    assert not res_df['kitchen_area'].isnull().any()
    assert res_df.loc[2, 'rooms'] == 2.5

def test_cleaner_inf_handling(sample_df):
    sample_df.loc[0, 'area'] = np.inf
    cleaner = DataCleaner()
    cleaner.fit(sample_df)
    res_df = cleaner.transform(sample_df)
    
    assert res_df.loc[0, 'area'] != np.inf
