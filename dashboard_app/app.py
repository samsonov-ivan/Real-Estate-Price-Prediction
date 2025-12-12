"""
App Initialization Module.
"""

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)

app.title = "Real Estate Auditor"
server = app.server