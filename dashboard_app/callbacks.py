"""
Callback Logic Module.
"""

import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback_context, Dash
from typing import Tuple, List, Any
from .state import state

def register_callbacks(app: Dash) -> None:
    """
    Registers all callback functions with the Dash application instance.

    Args:
        app (Dash): The initialized Dash application.
    """
    
    @app.callback(
        [Output("col-a-dropdown", "options"),
         Output("col-b-dropdown", "options"),
         Output("viz-column-dropdown", "options"),
         Output("col-a-dropdown", "value"),
         Output("viz-column-dropdown", "value")],
        [Input("url", "pathname")]
    )
    def init_controls(_: str) -> Tuple[List[dict], List[dict], List[dict], str, str]:
        """
        Populates dropdown menus based on the columns available in the state dataframe.
        """
        df = state.df
        if df.empty:
            return [], [], [], None, None
        
        num_cols = df.select_dtypes(include='number').columns
        options = [{"label": c, "value": c} for c in num_cols]
        all_options = [{"label": c, "value": c} for c in df.columns]
        
        default_num = num_cols[0] if not num_cols.empty else None
        default_viz = "price" if "price" in df.columns else df.columns[0]
        
        return options, options, all_options, default_num, default_viz

    @app.callback(
        [Output("metric-price", "children"),
         Output("metric-count", "children"),
         Output("metric-features", "children"),
         Output("dist-graph", "figure"),
         Output("map-graph", "figure"),
         Output("corr-graph", "figure"),
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
        prevent_initial_call=False
    )
    
    def update_all_metrics(
        viz_col: str, 
        n_clicks: int, 
        col_a: str, 
        op: str, 
        col_b: str, 
        new_name: str
    ) -> Any:
        """
        Main callback that handles both feature engineering and visual updates.

        1. Checks if the 'Create Feature' button was clicked.
        2. Applies mathematical operations to create a new column if valid.
        3. Updates KPIs (Average Price, Count).
        4. Updates Graphs (Histogram, Map, Heatmap).
        5. Refreshes dropdown options to include the new feature.
        """
        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        status_msg = ""
        df = state.df

        if df.empty:
             return "0", "0", "0", {}, {}, {}, "No Data", [], [], []

        if trigger == "create-btn" and n_clicks:
            if col_a and col_b and new_name and op:
                try:
                    if op == "add": 
                        df[new_name] = df[col_a] + df[col_b]
                    elif op == "sub": 
                        df[new_name] = df[col_a] - df[col_b]
                    elif op == "mul": 
                        df[new_name] = df[col_a] * df[col_b]
                    elif op == "div": 
                        df[new_name] = df[col_a] / (df[col_b] + 1e-9)
                    
                    state.df = df
                    status_msg = f"Feature '{new_name}' created!"
                except Exception as e:
                    status_msg = f"Error: {str(e)}"
            else:
                status_msg = "Fill all fields"

        avg_price = f"{df['price'].mean():,.0f} ₽" if 'price' in df.columns else "N/A"
        
        if viz_col and viz_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[viz_col]):
                fig_dist = px.histogram(
                    df, x=viz_col, nbins=50, 
                    title=f"Histogram of {viz_col}", 
                    template="plotly_white"
                )
            else:
                fig_dist = px.bar(
                    df[viz_col].value_counts().head(15), 
                    title=f"Top 15 {viz_col}", 
                    template="plotly_white"
                )
        else:
            fig_dist = px.histogram(template="plotly_white")

        if {'latitude', 'longitude', 'price', 'area'}.issubset(df.columns):
            plot_df = df.sample(min(len(df), 1000), random_state=42)
            fig_map = px.scatter_mapbox(
                plot_df, lat="latitude", lon="longitude", color="price", size="area",
                color_continuous_scale="Viridis", size_max=12, zoom=10, 
                mapbox_style="open-street-map", height=450
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        else:
            fig_map = px.scatter(title="Coordinates missing")

        corr = df.select_dtypes(include='number').corr()
        fig_corr = px.imshow(
            corr, text_auto=".1f", aspect="auto", 
            color_continuous_scale="RdBu_r"
        )

        num_cols = df.select_dtypes(include='number').columns
        opts_num = [{"label": c, "value": c} for c in num_cols]
        opts_all = [{"label": c, "value": c} for c in df.columns]

        return (avg_price, str(len(df)), str(df.shape[1]), 
                fig_dist, fig_map, fig_corr, status_msg,
                opts_num, opts_num, opts_all)
    