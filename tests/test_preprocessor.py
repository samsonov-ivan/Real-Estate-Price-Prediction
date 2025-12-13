import numpy as np
from src.preprocessor import DataCleaner, GeoDistanceTransformer

def test_geo_distance_transformer(sample_df):
    """
    Tests the GeoDistanceTransformer functionality.

    Args:
        sample_df (pd.DataFrame): Pytest fixture containing sample real estate data.
    """
    center = (55.7558, 37.6173)
    transformer = GeoDistanceTransformer(*center)
    
    res_df = transformer.transform(sample_df)
    
    assert 'distance_to_center' in res_df.columns
    assert not res_df['distance_to_center'].isnull().all()
    assert res_df.loc[0, 'distance_to_center'] < 10.0

def test_data_cleaner_fill(sample_df):
    """
    Tests the DataCleaner's ability to fill missing values.

    Args:
        sample_df (pd.DataFrame): Pytest fixture containing sample real estate data.
    """
    cleaner = DataCleaner()
    cleaner.fit(sample_df)
    res_df = cleaner.transform(sample_df)
    
    assert not res_df['rooms'].isnull().any()
    assert not res_df['kitchen_area'].isnull().any()
    assert res_df.loc[2, 'rooms'] == 2.5

def test_cleaner_inf_handling(sample_df):
    """
    Tests the handling of infinite values in the dataset.

    Args:
        sample_df (pd.DataFrame): Pytest fixture containing sample real estate data.
    """
    sample_df.loc[0, 'area'] = np.inf
    cleaner = DataCleaner()
    cleaner.fit(sample_df)
    res_df = cleaner.transform(sample_df)
    
    assert res_df.loc[0, 'area'] != np.inf
