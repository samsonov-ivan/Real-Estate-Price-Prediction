"""
Entry point to launch the Dashboard application.
"""

from dashboard_app.app import app
from dashboard_app.layout import create_layout
from dashboard_app.callbacks import register_callbacks
from dashboard_app.state import state

def main() -> None:
    """
    Main execution function for the dashboard.
    """
    app.layout = create_layout()
    
    register_callbacks(app)
    
    print("Initializing Dashboard State...")
    state.load_data("data/raw_sample.csv") 
    
    print("Starting Dash Server")
    app.run(debug=True, port=8050)

if __name__ == "__main__":
    main()
