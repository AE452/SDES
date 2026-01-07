import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import KDTree
#This is an interactive dashboard used for visualize the phase diagram from the simulation data
# Assuming the dataframe (df) is already loaded like this:
# df = pd.read_csv('your_dataframe.csv')  # Replace with actual data loading
df = pd.read_csv('./results/results_from_vectorized_script_raw.csv')

# Create a KDTree for fast nearest-neighbor search (based on alpha and lambda)
kdtree = KDTree(df[['alpha', 'lambda']].values)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define layout with sliders and heatmap plot
app.layout = html.Div([
    html.H1('Interactive Heatmap Dashboard'),

    # Alpha and Lambda sliders
    html.Div([
        dcc.Slider(
            id='alpha-slider',
            min=df['alpha'].min(),
            max=df['alpha'].max(),
            step=0.01,
            value=df['alpha'].min(),
            marks={i: str(round(i, 2)) for i in np.arange(df['alpha'].min(), df['alpha'].max(), 0.1)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        dcc.Slider(
            id='lambda-slider',
            min=df['lambda'].min(),
            max=df['lambda'].max(),
            step=0.01,
            value=df['lambda'].min(),
            marks={i: str(round(i, 2)) for i in np.arange(df['lambda'].min(), df['lambda'].max(), 0.1)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ]),

    # Graph for the heatmap
    dcc.Graph(id='heatmap-graph', style={'height': '800px', 'width': '100%'})

])

# Callback to update the heatmap based on alpha and lambda values
@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('alpha-slider', 'value'),
     Input('lambda-slider', 'value')]
)
def update_heatmap(alpha_val, lambda_val):
    # Use KDTree to find the closest (alpha, lambda) pair in the dataframe
    dist, idx = kdtree.query([alpha_val, lambda_val])
    closest_row = df.iloc[idx]
    alpha_val, lambda_val = closest_row['alpha'], closest_row['lambda']

    # Filter the dataframe based on the closest alpha, lambda pair
    filtered_df = df[(df['alpha'] == alpha_val) & (df['lambda'] == lambda_val)]

    # If no data found for the exact (alpha, lambda) pair, show data for the closest one
    if filtered_df.empty:
        filtered_df = df[(df['alpha'] == alpha_val) & (df['lambda'] == lambda_val)]

    # Define the 'xaxis', 'yaxis', and 'value' columns for the heatmap
    xaxis = 'e'  # or any other column you want to be on the x-axis
    yaxis = 'delta'  # or any other column for the y-axis
    value = 'P_AD'  # or any other column for the values to be visualized

    # Create the heatmap using Plotly
    heatmap_data = filtered_df.pivot_table(index=yaxis, columns=xaxis, values=value, aggfunc='mean')
    heatmap_data = heatmap_data.fillna(0)  # Fill NaN values with 0 if necessary

    # Generate the Plotly heatmap
    fig = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu',
        colorbar=dict(title=f'{value}'),
        hoverongaps=False
    ))

    fig.update_layout(
        title=f'Heatmap for Alpha={alpha_val}, Lambda={lambda_val}',
        xaxis_title=xaxis,
        yaxis_title=yaxis
    )

    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
