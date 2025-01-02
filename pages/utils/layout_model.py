import matplotlib.pyplot as plt
import plotly.graph_objects as go

import streamlit as st

def plot_epoch_vs_loss(model_name, df, train_col, val_col, title):

    # Create a line plot with Plotly
    fig = go.Figure()

    # Add traces for training losses
    fig.add_trace(go.Scatter(x=df['epoch'], y=df[train_col], mode='lines', line=dict(color="#BF0603"), name='Train'))

    # Add traces for validation losses
    fig.add_trace(go.Scatter(x=df['epoch'], y=df[val_col], mode='lines', line=dict(color="#003300"), name='Validation'))

    # Customize layout
    fig.update_layout(
        title={
            'text': f"{title} Loss vs Epoch",
            'x': 0.5,  # Center the title
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': '#333333'}    
        },
        xaxis_title="Epoch",
        yaxis_title="Loss",
        # legend_title="Loss Type",
        height=216,  # Reduce the height
        margin=dict(t=20, b=0),  # Reduce top and bottom margins
        template="plotly_white",
        legend=dict(
            x=0.75,  # Position legends inside the plot area
            y=0.90,
            bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent white background for legend
            # bordercolor="black",
            borderwidth=1
        )
    )

    # # # Display the Plotly chart in Streamlit
    st.plotly_chart(fig,)

    # # Create the plot
    # fig, ax = plt.subplots(figsize=(12, 6))

    # # Set the figure and axes background color
    # fig.patch.set_facecolor("#EEE5E9")  # Set figure background color
    # ax.set_facecolor("#EEE5E9")         # Set axes background color

    # # Plot the losses
    # ax.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss',)
    # ax.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss',)
    # ax.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss',)
    # ax.plot(df['epoch'], df['val/box_loss'], label='Validation Box Loss',)
    # ax.plot(df['epoch'], df['val/cls_loss'], label='Validation Class Loss',)
    # ax.plot(df['epoch'], df['val/dfl_loss'], label='Validation DFL Loss',)

    # # Customize labels and title
    # ax.set_xlabel('Epoch', fontsize=12)
    # ax.set_ylabel('Loss', fontsize=12)
    # ax.set_title(f'{model_name} Model Loss vs. Epoch', fontsize=14)
    # ax.legend(title="Loss Type", fontsize=8)

    # # Add grid with adjusted alpha for visibility
    # ax.grid(True, linestyle='--', alpha=1, color="#FFF6F3")

    # # Render the plot in Streamlit
    # st.pyplot(fig)