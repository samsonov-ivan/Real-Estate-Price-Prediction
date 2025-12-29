"""
Callback Logic Module.
"""

import pandas as pd
import numpy as np
import plotly.express as px
from dash import Input, Output, State, callback_context, Dash
from typing import Tuple, List, Any
from .state import state
from .components import summary_table, key_metrics_graph

MOSCOW_COORDS = {"lat": 55.7558, "lon": 37.6173}

def init_controls(_: str) -> Tuple[List[dict], List[dict], List[dict], str, str]:
    """Callback logic to initialize dropdowns."""
    df = state.df
    if df.empty:
        return [], [], [], None, None
    
    num_cols = df.select_dtypes(include='number').columns
    options = [{"label": c, "value": c} for c in num_cols]
    all_options = [{"label": c, "value": c} for c in df.columns]
    
    default_num = num_cols[0] if not num_cols.empty else None
    default_viz = "price" if "price" in df.columns else df.columns[0]
    
    return options, options, all_options, default_num, default_viz

def toggle_col_b(op: str) -> bool:
    """Callback logic to disable second column input."""
    unary_ops = ["square", "sqrt", "log", "zscore"]
    return op in unary_ops

def update_all_metrics(
    viz_col: str, n_clicks: int, col_a: str, op: str, col_b: str, new_name: str
) -> Any:
    """
    Main callback logic for updating graphs and engineering features.
    """
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "initial"
    status_msg = ""
    df = state.df

    if df.empty:
            return "0", "0", "0", {}, {}, {}, {}, {}, "No Data", [], [], []

    if trigger == "create-btn" and n_clicks:
        unary_ops = ["square", "sqrt", "log", "zscore"]
        is_valid = (col_a and new_name and op)
        if op not in unary_ops and not col_b:
            is_valid = False

        if is_valid:
            try:
                df = df.copy() 
                if op == "add": df[new_name] = df[col_a] + df[col_b]
                elif op == "sub": df[new_name] = df[col_a] - df[col_b]
                elif op == "mul": df[new_name] = df[col_a] * df[col_b]
                elif op == "div": df[new_name] = df[col_a] / (df[col_b] + 1e-9)
                elif op == "square": df[new_name] = df[col_a] ** 2
                elif op == "sqrt": df[new_name] = np.sqrt(np.abs(df[col_a]))
                elif op == "log": df[new_name] = np.log1p(np.abs(df[col_a]))
                elif op == "zscore":
                    mean = df[col_a].mean()
                    std = df[col_a].std()
                    df[new_name] = (df[col_a] - mean) / (std + 1e-9)
                
                state.df = df
                status_msg = f"Feature '{new_name}' created successfully."
            except Exception as e:
                status_msg = f"Error: {str(e)}"
        else:
            status_msg = "Please fill all required fields."

    avg_price = f"{df['price'].mean():,.0f} RUB" if 'price' in df.columns else "N/A"
    
    if viz_col and viz_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[viz_col]):
            fig_dist = px.histogram(
                df, x=viz_col, nbins=100,
                title=f"Histogram of {viz_col}", template="plotly_white"
            )
            upper_limit = df[viz_col].quantile(0.99)
            lower_limit = df[viz_col].quantile(0.01)
            fig_dist.update_xaxes(range=[lower_limit, upper_limit])
        else:
            fig_dist = px.bar(
                df[viz_col].value_counts().head(20), 
                title=f"Top 20 {viz_col}", template="plotly_white"
            )
        fig_dist.update_layout(height=600)
    else:
        fig_dist = px.histogram(template="plotly_white")

    if {'latitude', 'longitude', 'price', 'area'}.issubset(df.columns):
        plot_df = df.sample(min(len(df), 3000), random_state=42)
        upper_price_limit = df['price'].quantile(0.95)
        
        fig_map = px.scatter_mapbox(
            plot_df, lat="latitude", lon="longitude", 
            color="price", size="area",
            color_continuous_scale="Viridis", 
            range_color=[0, upper_price_limit],
            size_max=25,
            zoom=10,
            center=MOSCOW_COORDS,
            mapbox_style="open-street-map"
        )
        fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=800)
    else:
        fig_map = px.scatter(title="Coordinates missing")

    corr = df.select_dtypes(include='number').corr()
    fig_corr = px.imshow(
        corr, text_auto=".1f", aspect="auto", 
        color_continuous_scale="RdBu_r"
    )
    fig_corr.update_layout(height=700)

    summary_table_component = summary_table(df)
    key_metrics_fig = key_metrics_graph(df).figure

    num_cols = df.select_dtypes(include='number').columns
    opts_num = [{"label": c, "value": c} for c in num_cols]
    opts_all = [{"label": c, "value": c} for c in df.columns]

    return (avg_price, str(len(df)), str(df.shape[1]), 
            fig_dist, fig_map, fig_corr, 
            summary_table_component, key_metrics_fig, status_msg,
            opts_num, opts_num, opts_all)

def register_callbacks(app: Dash) -> None:
    """
    Registers the standalone functions to the Dash app.
    """
    
    app.callback(
        [Output("col-a-dropdown", "options"),
         Output("col-b-dropdown", "options"),
         Output("viz-column-dropdown", "options"),
         Output("col-a-dropdown", "value"),
         Output("viz-column-dropdown", "value")],
        [Input("url", "pathname")]
    )(init_controls)

    app.callback(
        Output("col-b-dropdown", "disabled"),
        Input("op-dropdown", "value")
    )(toggle_col_b)

    app.callback(
        [Output("metric-price", "children"),
         Output("metric-count", "children"),
         Output("metric-features", "children"),
         Output("dist-graph", "figure"),
         Output("map-graph", "figure"),
         Output("corr-graph", "figure"),
         Output("summary-table", "children"),
         Output("key-metrics-graph", "figure"),
         Output("creation-status", "children"),
         Output("col-a-dropdown", "options", allow_duplicate=True),
         Output("col-b-dropdown", "options", allow_duplicate=True),
         Output("viz-column-dropdown", "options", allow_duplicate=True)],
        [Input("viz-column-dropdown", "value"),
         Input("create-btn", "n_clicks")],
        [State("col-a-dropdown", "value"),
         State("op-dropdown", "value"),
         State("col-b-dropdown", "value"),
         State("new-col-name", "value")],
        prevent_initial_call=True
    )(update_all_metrics)
