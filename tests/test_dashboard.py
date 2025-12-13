"""
Tests for Dashboard application.
"""
import pytest
import pandas as pd
from unittest.mock import patch
from dashboard_app.state import AuditState, state
from dashboard_app.callbacks import update_all_metrics
import dash_bootstrap_components as dbc

from dashboard_app.app import app, server
from dashboard_app.components import (
    kpi_card, 
    feature_creation_panel, 
    correlation_heatmap, 
    map_visualization
)
from dashboard_app.layout import create_layout

@pytest.fixture
def mock_state_df(sample_df):
    """
    Populates the singleton state with test data.
    """
    state.df = sample_df.copy()
    state.original_df = sample_df.copy()
    yield
    state.df = pd.DataFrame()
    state.original_df = pd.DataFrame()

def test_audit_state_singleton():
    """Test that AuditState works as a singleton."""
    s1 = AuditState()
    s2 = AuditState()
    assert s1 is s2
    
    s1.df = pd.DataFrame({'a': [1]})
    assert not s2.df.empty
    assert s2.df.iloc[0]['a'] == 1

def test_app_initialization():
    """Test that the Dash app is initialized correctly."""
    assert app is not None
    assert server is not None
    assert app.title == "Real Estate Auditor"
    assert len(app.config.external_stylesheets) > 0

def test_kpi_card():
    """Test creation of a KPI card component."""
    card = kpi_card("Test Title", "test-id")
    
    assert isinstance(card, dbc.Card)
    card_body = card.children
    assert isinstance(card_body, dbc.CardBody)
    
    found_id = False
    for child in card_body.children:
        if getattr(child, "id", None) == "test-id":
            found_id = True
            break
    assert found_id, "Value element with specific ID not found in card"

def test_feature_creation_logic(mock_state_df):
    """
    Tests the logic inside the main callback for creating a new feature.
    """
    with patch("dashboard_app.callbacks.callback_context") as mock_ctx:
        mock_ctx.triggered = [{"prop_id": "create-btn.n_clicks"}]
        
        output = update_all_metrics(
            "price", 1, 
            "area", "square", None, "area_sq"
        )
        
        assert "area_sq" in state.df.columns
        assert state.df["area_sq"].iloc[0] == 2500.0
        
        assert len(output) == 10
        assert "successfully" in output[6]

def test_singleton_pattern():
    s1 = AuditState()
    s2 = AuditState()
    assert s1 is s2
    
    s1.df = pd.DataFrame({"test": [1]})
    assert not s2.df.empty
    assert s2.df.iloc[0]["test"] == 1



def test_callback_visualization_output(mock_state_df):
    """
    Tests that the callback returns valid Plotly figures.
    """
    with patch("dashboard_app.callbacks.callback_context") as mock_ctx:
        mock_ctx.triggered = [{"prop_id": "viz-column-dropdown.value"}]
        
        output = update_all_metrics(
            "price", 0, 
            None, None, None, None
        )
        
        fig_dist = output[3]
        fig_map = output[4]
        fig_corr = output[5]
        
        assert hasattr(fig_dist, "layout") or isinstance(fig_dist, dict)
        assert hasattr(fig_map, "layout") or isinstance(fig_map, dict)
        assert "RUB" in output[0]

def test_feature_creation_error_handling(mock_state_df):
    """Test error handling when creating invalid features."""
    with patch("dashboard_app.callbacks.callback_context") as mock_ctx:
        mock_ctx.triggered = [{"prop_id": "create-btn.n_clicks"}]
        
        output = update_all_metrics(
            "price", 1, 
            "area", "div", "non_existent_col", "bad_feat"
        )
        
        assert "Please fill" in output[6] or "Error" in output[6]
