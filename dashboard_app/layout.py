"""
Layout Definition Module.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from .components import kpi_card, feature_creation_panel

def create_layout() -> html.Div:
    """
    Constructs the main layout hierarchy of the application.

    Returns:
        html.Div: The root Dash component containing the entire UI.
    """
    return html.Div([
        dcc.Location(id="url"),
        
        dbc.NavbarSimple(
            brand="Real Estate Auditor Pro",
            brand_href="#",
            color="primary",
            dark=True,
            className="mb-4 shadow"
        ),

        dbc.Container([
            dbc.Row([
                dbc.Col([
                    feature_creation_panel(),
                    dbc.Card([
                        dbc.CardHeader("Settings"),
                        dbc.CardBody([
                            html.P("Analyze specific segments.", className="small text-muted"),
                            html.Label("Visualize Feature:"),
                            dcc.Dropdown(id="viz-column-dropdown", className="mb-2")
                        ])
                    ], className="shadow-sm")
                ], md=3),

                dbc.Col([
                    dbc.Row([
                        dbc.Col(kpi_card("Average Price", "metric-price"), md=4),
                        dbc.Col(kpi_card("Total Objects", "metric-count"), md=4),
                        dbc.Col(kpi_card("Active Features", "metric-features"), md=4),
                    ], className="mb-4"),

                    dbc.Tabs([
                        dbc.Tab(label="Distribution Analysis", tab_id="tab-dist", children=[
                            dbc.Card(dbc.CardBody([
                                dcc.Graph(id="dist-graph")
                            ]), className="mt-3 border-0 shadow-sm")
                        ]),
                        dbc.Tab(label="Geo & Price Map", tab_id="tab-map", children=[
                             dbc.Card(dbc.CardBody([
                                dcc.Graph(id="map-graph")
                            ]), className="mt-3 border-0 shadow-sm")
                        ]),
                        dbc.Tab(label="Correlations", tab_id="tab-corr", children=[
                             dbc.Card(dbc.CardBody([
                                dcc.Graph(id="corr-graph")
                            ]), className="mt-3 border-0 shadow-sm")
                        ]),
                        dbc.Tab(label="Summary Statistics", tab_id="tab-summary", children=[
                            dbc.Card(dbc.CardBody([
                                html.Div(id="summary-table")
                            ]), className="mt-3 border-0 shadow-sm")
                        ]),
                        dbc.Tab(label="Key Metrics", tab_id="tab-key-metrics", children=[
                            dbc.Card(dbc.CardBody([
                                dcc.Graph(id="key-metrics-graph")
                            ]), className="mt-3 border-0 shadow-sm")
                        ]),
                    ], id="tabs", active_tab="tab-dist")
                    
                ], md=9)
            ])
        ], fluid=True)
    ], className="bg-light min-vh-100")
