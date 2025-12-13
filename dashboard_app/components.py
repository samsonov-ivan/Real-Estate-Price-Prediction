"""
UI Components Module.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
import pandas as pd

MOSCOW_COORDS = {"lat": 55.7558, "lon": 37.6173}


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

    Returns:
        dbc.Card: A Bootstrap card component for the sidebar.
    """
    return dbc.Card([
        dbc.CardHeader("Feature Lab"),
        dbc.CardBody([
            html.Label("Column A (Main)", className="fw-bold"),
            dcc.Dropdown(id="col-a-dropdown", className="mb-2", clearable=False),
            
            html.Label("Operation", className="fw-bold"),
            dcc.Dropdown(
                id="op-dropdown",
                options=[
                    {"label": "Add (A + B)", "value": "add"},
                    {"label": "Subtract (A - B)", "value": "sub"},
                    {"label": "Multiply (A * B)", "value": "mul"},
                    {"label": "Divide (A / B)", "value": "div"},
                    {"label": "Square (A^2)", "value": "square"},
                    {"label": "Square Root (sqrt(A))", "value": "sqrt"},
                    {"label": "Logarithm (log(1+A))", "value": "log"},
                    {"label": "Standardize (Z-Score)", "value": "zscore"},
                ],
                value="add",
                className="mb-2",
                clearable=False
            ),
            
            html.Label("Column B (Optional)", className="fw-bold"),
            dcc.Dropdown(
                id="col-b-dropdown", 
                className="mb-2", 
                clearable=False,
                placeholder="Select second column..."
            ),
            
            html.Label("New Feature Name", className="fw-bold"),
            dbc.Input(id="new-col-name", placeholder="e.g. price_squared", className="mb-3"),
            
            dbc.Button("Apply Transformation", id="create-btn", color="primary", className="w-100"),
            html.Div(id="creation-status", className="mt-2 small")
        ])
    ], className="mb-4 shadow-sm")


def correlation_heatmap(df: pd.DataFrame) -> dcc.Graph:
    """
    Generates a correlation heatmap graph component.
    """
    corr = df.select_dtypes(include='number').corr()
    
    fig = px.imshow(
        corr, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="RdBu_r",
        title="Numeric Feature Correlation"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=700)
    return dcc.Graph(id="corr-graph", figure=fig)


def map_visualization(df: pd.DataFrame) -> dcc.Graph:
    """
    Generates a map visualization graph component with Moscow focus and auto-scaled colors.

    Args:
        df (pd.DataFrame): The dataframe containing 'latitude', 'longitude', and 'price'.

    Returns:
        dcc.Graph: A Dash Graph component with the map.
    """
    if 'latitude' in df.columns and 'longitude' in df.columns:
        plot_df = df.sample(min(len(df), 3000), random_state=42)
        
        upper_price_limit = df['price'].quantile(0.95)
        
        fig = px.scatter_mapbox(
            plot_df, 
            lat="latitude", lon="longitude", 
            color="price", 
            size="area",
            color_continuous_scale=px.colors.cyclical.IceFire, 
            range_color=[0, upper_price_limit],
            size_max=20,
            zoom=10,
            center=MOSCOW_COORDS,
            mapbox_style="open-street-map",
            title="Real Estate Geography (Color scaled to 95th percentile)"
        )
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=800)
    else:
        fig = px.scatter(title="Coordinates (latitude/longitude) not found in dataset")
        
    return dcc.Graph(id="map-graph", figure=fig)

def summary_table(df: pd.DataFrame) -> dash_table.DataTable:
    """
    Generates a summary statistics table.
    """
    summary = df.describe().reset_index().round(2)
    return dash_table.DataTable(
        id='summary-table-inner',
        data=summary.to_dict('records'),
        columns=[{"name": i, "id": i} for i in summary.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'minWidth': '100px'},
    )


def key_metrics_graph(df: pd.DataFrame) -> dcc.Graph:
    """
    Generates a bar chart for key metrics like average price, median area, etc.
    """
    if 'price' in df.columns and 'area' in df.columns:
        metrics = {
            'Avg Price': df['price'].mean(),
            'Median Price': df['price'].median(),
            'Avg Area': df['area'].mean(),
            'Median Area': df['area'].median(),
            'Count': len(df)
        }
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        fig = px.bar(
            metrics_df, x='Metric', y='Value',
            title="Key Real Estate Metrics",
            template="plotly_white"
        )
        fig.update_layout(height=600)
    else:
        fig = px.bar(title="Required columns (price/area) missing")
    return dcc.Graph(id="key-metrics-graph", figure=fig)
