import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from pathlib import Path
from contextlib import redirect_stdout
from ocr import PerceptronOCR as ocr

# Page configuration
st.set_page_config(
    page_title="OCR Neural Network Trainer", page_icon="🧠", layout="wide"
)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "training_history" not in st.session_state:
    st.session_state.training_history = {"loss": [], "accuracy": [], "epochs": []}
if "drawing_grid" not in st.session_state:
    st.session_state.drawing_grid = np.zeros((7, 5))


def create_drawing_interface():
    """Create a 5x7 pixel drawing interface"""
    st.subheader("Draw a Character (5x7 pixels)")

    # Create buttons for each pixel
    cols = st.columns(5)

    for row in range(7):
        for col in range(5):
            with cols[col]:
                # Create a button for each pixel
                if st.button(
                    "🟩" if st.session_state.drawing_grid[row, col] == 1 else "⬜",
                    key=f"pixel_{row}_{col}",
                    help=f"Pixel ({row}, {col})",
                ):
                    # Toggle pixel
                    st.session_state.drawing_grid[row, col] = (
                        1 - st.session_state.drawing_grid[row, col]
                    )
                    st.rerun()

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Grid"):
            st.session_state.drawing_grid = np.zeros((7, 5))
            st.rerun()

    with col2:
        if st.button("Fill Grid"):
            st.session_state.drawing_grid = np.ones((7, 5))
            st.rerun()

    # Show current grid as array
    with st.expander("Current Grid Array"):
        st.text("7x5 Grid:")
        st.text(str(st.session_state.drawing_grid.astype(int)))
        st.text("Flattened (35 elements):")
        flattened = st.session_state.drawing_grid.flatten()
        st.text(str(flattened.astype(int)))


def train_model(model, **kwargs):
    """Train the model and capture output"""
    output_buffer = io.StringIO()

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Capture training output
    with redirect_stdout(output_buffer):
        try:
            # Train the model
            model.train(**kwargs)

            # Get training history from the model after training
            if hasattr(model, "train_history") and model.train_history:
                history_df = pd.DataFrame(model.train_history)
                st.session_state.training_history = {
                    "epochs": history_df["epoch"].tolist(),
                    "loss": history_df["loss"].tolist(),
                    "accuracy": history_df["accuracy"].tolist(),
                }

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            return False

    progress_bar.progress(1.0)
    status_text.text("Training completed!")

    # Display training output
    training_output = output_buffer.getvalue()
    if training_output:
        with st.expander("Training Output"):
            st.text(training_output)

    return True


def display_training_plots():
    """Display training loss and accuracy plots using model history"""
    if not st.session_state.training_history["loss"]:
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Training Loss", "Training Accuracy"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )

    # Loss plot
    fig.add_trace(
        go.Scatter(
            x=st.session_state.training_history["epochs"],
            y=st.session_state.training_history["loss"],
            mode="lines+markers",
            name="Loss",
            line=dict(color="red", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=st.session_state.training_history["epochs"],
            y=st.session_state.training_history["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Training Metrics Over Time",
        showlegend=False,
        height=400,
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)


def display_training_history_table():
    """Display training history as a dataframe table"""
    if (
        st.session_state.model
        and hasattr(st.session_state.model, "train_history")
        and st.session_state.model.train_history
    ):
        st.subheader("Training History Details")

        # Get history dataframe from model
        history_df = pd.DataFrame(st.session_state.model.train_history)

        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Epochs", len(history_df))
        with col2:
            st.metric("Final Loss", f"{history_df['loss'].iloc[-1]:.4f}")
        with col3:
            st.metric("Final Accuracy", f"{history_df['accuracy'].iloc[-1]:.4f}")
        with col4:
            st.metric("Best Accuracy", f"{history_df['accuracy'].max():.4f}")

        # Show the dataframe
        with st.expander("View Full Training History"):
            st.dataframe(history_df.round(4), use_container_width=True, height=300)

        # Download button for history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Training History as CSV",
            data=csv,
            file_name="training_history.csv",
            mime="text/csv",
        )


def main():
    st.title("🧠 OCR Neural Network Trainer")
    st.markdown("---")

    # Sidebar for model parameters
    st.sidebar.title("Training Parameters")

    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        options=[
            "Upload CSV File",
            "Use CSV Digits Dataset",
            "Use More Digits Dataset",
        ],
        help="Upload your own CSV file or use the digits CSV dataset",
    )

    file_path = None

    if data_source == "Upload CSV File":
        # File upload
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file is not None:
            # Use pathlib to create a temp directory in the current working directory
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)

            # Save uploaded file temporarily
            file_extension = uploaded_file.name.split(".")[-1].lower()
            file_path = temp_dir / f"temp_data.{file_extension}"

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            st.sidebar.success("✅ File uploaded successfully!")

    elif data_source == "Use CSV Digits Dataset":
        # Check if CSV dataset exists

        csv_path = Path("./data/digits_data.csv")

        if csv_path.exists():
            file_path = str(csv_path)
            st.sidebar.success("✅ CSV digits dataset found!")
            st.sidebar.info(f"Using: {csv_path}")
        else:
            st.sidebar.error("❌ CSV digits dataset not found!")
            st.sidebar.error("Please ensure ./digits_data.csv exists or upload a file.")

    elif data_source == "Use More Digits Dataset":
        # Check if More Digits dataset exists
        csv_path = Path("./data/more_digits_data.csv")

        if csv_path.exists():
            file_path = str(csv_path)
            st.sidebar.success("✅ More digits dataset found!")
            st.sidebar.info(f"Using: {csv_path}")
        else:
            st.sidebar.error("❌ More digits dataset not found!")
            st.sidebar.error(
                "Please ensure ./more_digits_data.csv exists or upload a file."
            )

    # Initialize model if file path is available
    if file_path is not None and st.session_state.model is None:
        try:
            st.session_state.model = ocr(file_path)
            st.sidebar.success("✅ Model initialized successfully!")
            st.sidebar.write(f"📊 Features shape: {st.session_state.model.X.shape}")
            st.sidebar.write(f"🏷️ Labels shape: {st.session_state.model.Y.shape}")
            st.sidebar.write(
                f"🔢 Input features: {st.session_state.model.X.shape[1]} (row + column sums)"
            )
            st.sidebar.write(
                f"🎯 Target range: {st.session_state.model.Y.min():.0f} - {st.session_state.model.Y.max():.0f} (ASCII values)"
            )
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing model: {str(e)}")
            st.sidebar.error("Please check your file format.")
            return
    elif file_path is None:
        if data_source == "Upload CSV File":
            st.info("📁 Please upload a CSV file to begin.")
        elif data_source == "Use CSV Digits Dataset":
            st.error(
                "📁 CSV digits dataset not found. Please upload a file or ensure ./digits_data.csv exists."
            )
        return

    # Training parameters
    st.sidebar.subheader("Neural Network Parameters")

    # Auto-detect input size based on format
    if st.session_state.model is not None:
        default_input_size = st.session_state.model.X.shape[1]
        st.sidebar.info(
            f"CSV Mode: Using {default_input_size} features (row + column sums)"
        )
        default_hidden_size = 16
    else:
        default_input_size = 12
        default_hidden_size = 16

    input_size = st.sidebar.number_input(
        "Input Size", value=default_input_size, min_value=1, max_value=1000
    )
    hidden_size = st.sidebar.number_input(
        "Hidden Size", value=default_hidden_size, min_value=1, max_value=1000
    )
    learning_rate = st.sidebar.number_input(
        "Learning Rate",
        value=0.1,
        min_value=0.001,
        max_value=1.0,
        step=0.001,
        format="%.3f",
    )
    epochs = st.sidebar.number_input("Epochs", value=1000, min_value=1, max_value=10000)

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        # Training section
        st.subheader("Model Training")

        if st.button(
            "🚀 Train Model", type="primary", disabled=(st.session_state.model is None)
        ):
            with st.spinner("Training model..."):
                success = train_model(
                    st.session_state.model,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                )
                if success:
                    st.session_state.is_trained = True
                    st.success("✅ Model trained successfully!")

                    # Show sample predictions
                    st.subheader("Sample Predictions")
                    output_buffer = io.StringIO()
                    with redirect_stdout(output_buffer):
                        st.session_state.model.sample_predict()

                    sample_output = output_buffer.getvalue()
                    if sample_output:
                        st.text(sample_output)

        # Display training metrics if available
        if st.session_state.is_trained and st.session_state.training_history["loss"]:
            st.subheader("Training Metrics")
            final_loss = (
                st.session_state.training_history["loss"][-1]
                if st.session_state.training_history["loss"]
                else 0
            )
            final_accuracy = (
                st.session_state.training_history["accuracy"][-1]
                if st.session_state.training_history["accuracy"]
                else 0
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Final Loss", f"{final_loss:.4f}")
            with metric_col2:
                st.metric("Final Accuracy", f"{final_accuracy:.4f}")
            with metric_col3:
                st.metric("Epochs", epochs)

    with col2:
        # Inference section
        st.subheader("Character Inference")

        if not st.session_state.is_trained:
            st.info("🔒 Please train the model first to enable inference.")
        else:
            # Drawing interface
            create_drawing_interface()

            # Inference button
            if st.button(
                "🔍 Predict Character", disabled=not st.session_state.is_trained
            ):
                if st.session_state.is_trained and st.session_state.model is not None:
                    try:
                        # Only CSV format: compute 12-element feature vector
                        grid = st.session_state.drawing_grid
                        row_sums = np.sum(grid, axis=1)
                        col_sums = np.sum(grid, axis=0)
                        input_features = np.concatenate([row_sums, col_sums]).reshape(
                            1, -1
                        )

                        # Make prediction (regression)
                        prediction = st.session_state.model.predict(input_features)
                        pred_ascii = int(np.round(prediction[0]))
                        predicted_char = (
                            chr(pred_ascii) if 32 <= pred_ascii <= 126 else "?"
                        )
                        raw_output = float(prediction[0])
                        st.success(f"""
                                   🎯 Predicted Character: **{predicted_char}**
                                   🖥 ASCII Value: **{pred_ascii}**
                                   📊 Raw Output: **{raw_output:.2f}**
                                   """)

                        # Show feature vector
                        st.subheader("Feature Analysis")
                        st.write(f"Row sums (7 values): {row_sums}")
                        st.write(f"Column sums (5 values): {col_sums}")
                        st.write(
                            f"Combined features (12 values): {np.concatenate([row_sums, col_sums])}"
                        )
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")


# Run the app
if __name__ == "__main__":
    main()
