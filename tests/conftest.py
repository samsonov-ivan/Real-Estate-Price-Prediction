import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_df():
    """Creates a small dataframe for testing."""
    data = {
        'latitude': [55.75, 55.76, np.nan],
        'longitude': [37.61, 37.62, 37.61],
        'area': [50.0, 100.0, 75.0],
        'kitchen_area': [10.0, np.nan, 15.0],
        'rooms': [2, 3, np.nan],
        'price': [100, 200, 150],
        'building_type': ['1', '2', '1'],
        'id_region': ['77', '77', '50']
    }
    return pd.DataFrame(data)
