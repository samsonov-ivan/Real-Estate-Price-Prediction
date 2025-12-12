"""
UI Components Module.

This module contains functions that return reusable Dash/Bootstrap components,
such as KPI cards, the feature creation sidebar, and graph wrappers.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd


def kpi_card(title: str, value_id: str) -> dbc.Card:
    """
    Creates a standardized KPI card.

    Args:
        title (str): The title to display on the card.
        value_id (str): The Dash ID for the value element (used in callbacks).

    Returns:
        dbc.Card: A Bootstrap card component containing the title and dynamic value.
    """
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted mb-2"),
            html.H2("Loading...", id=value_id, className="card-title text-primary"),
        ]),
        className="shadow-sm h-100 border-0"
    )


def feature_creation_panel() -> dbc.Card:
    """
    Creates the Feature Engineering sidebar panel.

    This panel includes dropdowns for selecting columns and operations
    to generate new features dynamically.

    Returns:
        dbc.Card: A Bootstrap card component for the sidebar.
    """
    return dbc.Card([
        dbc.CardHeader("Feature Lab"),
        dbc.CardBody([
            html.Label("Column A", className="fw-bold"),
            dcc.Dropdown(id="col-a-dropdown", className="mb-2", clearable=False),
            
            html.Label("Operation", className="fw-bold"),
            dcc.Dropdown(
                id="op-dropdown",
                options=[
                    {"label": "Add", "value": "add"},
                    {"label": "Subtract", "value": "sub"},
                    {"label": "Multiply", "value": "mul"},
                    {"label": "Divide", "value": "div"},
                ],
                value="add",
                className="mb-2",
                clearable=False
            ),
            
            html.Label("Column B", className="fw-bold"),
            dcc.Dropdown(id="col-b-dropdown", className="mb-2", clearable=False),
            
            html.Label("New Feature Name", className="fw-bold"),
            dbc.Input(id="new-col-name", placeholder="e.g. price_per_meter", className="mb-3"),
            
            dbc.Button("Apply Transformation", id="create-btn", color="primary", className="w-100"),
            html.Div(id="creation-status", className="mt-2 small")
        ])
    ], className="mb-4 shadow-sm")


def correlation_heatmap(df: pd.DataFrame) -> dcc.Graph:
    """
    Generates a correlation heatmap graph component.

    Args:
        df (pd.DataFrame): The dataframe containing the data.

    Returns:
        dcc.Graph: A Dash Graph component with the heatmap.
    """
    corr = df.select_dtypes(include='number').corr()
    
    fig = px.imshow(
        corr, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="RdBu_r",
        title="Numeric Feature Correlation"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return dcc.Graph(id="corr-graph", figure=fig)


def map_visualization(df: pd.DataFrame) -> dcc.Graph:
    """
    Generates a map visualization graph component.

    Args:
        df (pd.DataFrame): The dataframe containing 'latitude', 'longitude', and 'price'.

    Returns:
        dcc.Graph: A Dash Graph component with the map.
    """
    if 'latitude' in df.columns and 'longitude' in df.columns:
        plot_df = df.head(500)
        
        fig = px.scatter_mapbox(
            plot_df, 
            lat="latitude", lon="longitude", 
            color="price", size="area",
            color_continuous_scale=px.colors.cyclical.IceFire, 
            size_max=15, zoom=9, 
            mapbox_style="open-street-map",
            title="Real Estate Geography"
        )
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    else:
        fig = px.scatter(title="Coordinates (latitude/longitude) not found in dataset")
        
    return dcc.Graph(id="map-graph", figure=fig, style={"height": "400px"})
