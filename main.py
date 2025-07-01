import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from pathlib import Path
from contextlib import redirect_stdout

# Import all three OCR models
from glorot import GlorotOCR
from kaiming import KaimingOCR
from ocr import PerceptronOCR


# Page configuration
st.set_page_config(
    page_title="OCR Neural Network Trainer", page_icon="🧠", layout="wide"
)

# Initialize session state
if "models" not in st.session_state:
    st.session_state.models = {}  # Store all initialized models
if "training_results" not in st.session_state:
    st.session_state.training_results = {}  # Store results for each model
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False
if "drawing_grid" not in st.session_state:
    st.session_state.drawing_grid = np.zeros((7, 5))


def create_drawing_interface():
    """Create a 5x7 pixel drawing interface using dataframe with checkboxes"""
    st.subheader("Draw a Character (5x7 pixels)")

    # Create a dataframe for the drawing grid
    # Convert numpy array to dataframe with boolean values for checkboxes
    grid_df = pd.DataFrame(
        st.session_state.drawing_grid.astype(bool),
        columns=[f"Col_{i}" for i in range(5)],
        index=[f"Row_{i}" for i in range(7)]
    )

    # Use data_editor with checkboxes (without automatic updates)
    edited_df = st.data_editor(
        grid_df,
        use_container_width=True,
        hide_index=False,
        column_config={
            f"Col_{i}": st.column_config.CheckboxColumn(
                f"Col {i}",
                help=f"Column {i}",
                default=False,
            ) for i in range(5)
        },
        key="drawing_grid_editor",
        disabled=False
    )

    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Update Grid", type="primary", help="Apply your checkbox changes to the drawing grid"):
            st.session_state.drawing_grid = edited_df.values.astype(int)
            st.success("✅ Grid updated!")
            st.rerun()

    with col2:
        if st.button("🧹 Clear Grid"):
            st.session_state.drawing_grid = np.zeros((7, 5))
            st.rerun()

    with col3:
        if st.button("🟩 Fill Grid"):
            st.session_state.drawing_grid = np.ones((7, 5))
            st.rerun()

    # Show current grid as array
    with st.expander("Current Grid Array"):
        st.text("7x5 Grid:")
        st.text(str(st.session_state.drawing_grid.astype(int)))
        st.text("Flattened (35 elements):")
        flattened = st.session_state.drawing_grid.flatten()
        st.text(str(flattened.astype(int)))


def train_model(model_name, model_instance, **kwargs):
    """Train a single model and capture output and history."""
    output_buffer = io.StringIO()
    success = False
    training_history = {"loss": [], "accuracy": [], "epochs": []}

    try:
        # Before training, ensure the model's history is clear for a fresh run
        if hasattr(model_instance, 'train_history'):
            model_instance.train_history = [] 

        with redirect_stdout(output_buffer):
            model_instance.train(**kwargs)

        if hasattr(model_instance, "train_history") and model_instance.train_history:
            history_df = pd.DataFrame(model_instance.train_history)
            training_history = {
                "epochs": history_df["epoch"].tolist(),
                "loss": history_df["loss"].tolist(),
                "accuracy": history_df["accuracy"].tolist(),
            }
        success = True
    except Exception as e:
        st.error(f"Training {model_name} failed: {str(e)}")

    return success, output_buffer.getvalue(), training_history


def display_training_comparison(training_results):
    """Display training results in a comparative table and plots."""
    if not training_results:
        return

    st.subheader("Training Results Comparison")

    # Prepare data for the table
    table_data = []
    for model_name, result in training_results.items():
        final_loss = result["training_history"]["loss"][-1] if result["training_history"]["loss"] else np.nan
        final_accuracy = result["training_history"]["accuracy"][-1] if result["training_history"]["accuracy"] else np.nan
        best_accuracy = max(result["training_history"]["accuracy"]) if result["training_history"]["accuracy"] else np.nan
        table_data.append(
            {
                "Model": model_name,
                "Final Loss": f"{final_loss:.4f}",
                "Final Accuracy": f"{final_accuracy:.4f}",
                "Best Accuracy": f"{best_accuracy:.4f}"
            }
        )
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # Plotting Loss and Accuracy over Epochs for all models
    st.subheader("Training Metrics Over Time (Comparison)")

    fig_loss = go.Figure()
    fig_accuracy = go.Figure()

    for model_name, result in training_results.items():
        history = result["training_history"]
        if history["epochs"]:
            fig_loss.add_trace(
                go.Scatter(
                    x=history["epochs"],
                    y=history["loss"],
                    mode="lines",
                    name=f"{model_name} Loss",
                )
            )
            fig_accuracy.add_trace(
                go.Scatter(
                    x=history["epochs"],
                    y=history["accuracy"],
                    mode="lines",
                    name=f"{model_name} Accuracy",
                )
            )

    fig_loss.update_layout(
        title="Training Loss Comparison",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
    )
    fig_accuracy.update_layout(
        title="Training Accuracy Comparison",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode="x unified",
    )

    st.plotly_chart(fig_loss, use_container_width=True)
    st.plotly_chart(fig_accuracy, use_container_width=True)

    # Display full training history dataframes in expanders
    st.subheader("Detailed Training History")
    for model_name, result in training_results.items():
        if result["training_history"]["epochs"]:
            with st.expander(f"View {model_name} Training History"):
                history_df = pd.DataFrame(result["training_history"])
                st.dataframe(history_df.round(4), use_container_width=True, height=200)
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {model_name} History as CSV",
                    data=csv,
                    file_name=f"{model_name}_training_history.csv",
                    mime="text/csv",
                    key=f"download_{model_name}"
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

    # Track previous data source and file path
    if "prev_data_source" not in st.session_state:
        st.session_state.prev_data_source = data_source
    if "prev_file_path" not in st.session_state:
        st.session_state.prev_file_path = None

    # Detect data source or file path change and reset models if needed
    if st.session_state.prev_data_source != data_source or (
        file_path is not None and st.session_state.prev_file_path != str(file_path)
    ):
        st.session_state.models = {}
        st.session_state.training_results = {}
        st.session_state.is_trained = False
        st.session_state.prev_data_source = data_source
        st.session_state.prev_file_path = (
            str(file_path) if file_path is not None else None
        )

    # Initialize all models if file path is available and models are not initialized
    # OR if file_path has changed, re-initialize them
    if file_path is not None and (not st.session_state.models or st.session_state.prev_file_path != str(file_path)):
        try:
            st.session_state.models["PerceptronOCR"] = PerceptronOCR(file_path)
            st.session_state.models["GlorotOCR"] = GlorotOCR(file_path)
            st.session_state.models["KaimingOCR"] = KaimingOCR(file_path)

            st.sidebar.success("✅ All models initialized successfully!")
            # Display info from the first initialized model (they all preprocess data similarly)
            if "PerceptronOCR" in st.session_state.models:
                model_ref = st.session_state.models["PerceptronOCR"]
                st.sidebar.write(f"📊 Features shape: {model_ref.X.shape}")
                st.sidebar.write(f"🏷️ Labels shape: {model_ref.Y.shape}")
                st.sidebar.write(
                    f"🔢 Input features: {model_ref.X.shape[1]} (row + column sums)"
                )
                st.sidebar.write(
                    f"🎯 Target range: {model_ref.Y.min():.0f} - {model_ref.Y.max():.0f} (ASCII values)"
                )
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing models: {str(e)}")
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
    default_input_size = 12
    default_hidden_size = 16
    if "PerceptronOCR" in st.session_state.models:
        default_input_size = st.session_state.models["PerceptronOCR"].X.shape[1]
        st.sidebar.info(
            f"CSV Mode: Using {default_input_size} features (row + column sums)"
        )

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
    epochs = st.sidebar.number_input("Epochs", value=10000, min_value=1, max_value=500000)

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        # Training section
        st.subheader("Model Training")

        if st.button(
            "🚀 Train All Models", type="primary", disabled=(not st.session_state.models)
        ):
            st.session_state.training_results = {}
            all_trained_successfully = True

            # Re-initialize models before training if they already exist
            if file_path is not None:
                try:
                    st.session_state.models["PerceptronOCR"] = PerceptronOCR(file_path)
                    st.session_state.models["GlorotOCR"] = GlorotOCR(file_path)
                    st.session_state.models["KaimingOCR"] = KaimingOCR(file_path)
                    st.info("🔄 Re-initialized all models for a fresh training run.")
                except Exception as e:
                    st.error(f"❌ Error re-initializing models: {str(e)}")
                    st.session_state.is_trained = False
                    return # Stop if re-initialization fails

            for model_name, model_instance in st.session_state.models.items():
                st.info(f"Training {model_name}...")
                progress_bar = st.progress(0, text=f"Training {model_name}...")
                
                # Use a specific key for each model's spinner to avoid conflicts
                with st.spinner(f"Training {model_name} model..."):
                    success, output, history = train_model(
                        model_name,
                        model_instance,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        learning_rate=learning_rate,
                        epochs=epochs,
                    )
                    st.session_state.training_results[model_name] = {
                        "output": output,
                        "training_history": history,
                    }
                    if not success:
                        all_trained_successfully = False
                
                progress_bar.progress(1.0, text=f"{model_name} training complete!")
                st.success(f"✅ {model_name} trained successfully!")
                with st.expander(f"View {model_name} Training Output"):
                    st.text(output)


            st.session_state.is_trained = all_trained_successfully
            if all_trained_successfully:
                st.success("✅ All models trained successfully!")
            else:
                st.warning("Training completed with some failures.")
            st.rerun() # Rerun to update the display


        # Display training comparison table and plots if results exist
        if st.session_state.is_trained and st.session_state.training_results:
            display_training_comparison(st.session_state.training_results)
            st.markdown("---")
            st.subheader("Sample Predictions from Trained Models")
            for model_name, model_instance in st.session_state.models.items():
                with st.expander(f"Sample Predictions for {model_name}"):
                    sample_output_buffer = io.StringIO()
                    with redirect_stdout(sample_output_buffer):
                        model_instance.sample_predict()
                    st.text(sample_output_buffer.getvalue())


    with col2:
        # Inference section
        st.subheader("Character Inference")

        if not st.session_state.is_trained:
            st.info("🔒 Please train the models first to enable inference.")
        else:
            # Create tabs for different input methods
            tab1, tab2 = st.tabs(["🎨 Drawing Grid", "⌨️ Custom Input"])
            
            with tab1:
                # Drawing interface
                create_drawing_interface()
                
                # Inference button for drawing grid
                if st.button(
                    "🔍 Predict Character from Grid", 
                    disabled=not st.session_state.is_trained,
                    key="predict_grid"
                ):
                    if st.session_state.is_trained and st.session_state.models:
                        st.subheader("Prediction Results - Drawing Grid")

                        # Prediction for drawn grid
                        try:
                            grid = st.session_state.drawing_grid
                            
                            # Display the drawn grid
                            st.text("Drawn Grid (7x5):")
                            for row in grid.astype(int):
                                st.write("".join(["🟩" if p == 1 else "⬜" for p in row]))
                            
                            row_sums = np.sum(grid, axis=1)
                            col_sums = np.sum(grid, axis=0)
                            input_features_drawn = np.concatenate([row_sums, col_sums]).reshape(
                                1, -1
                            )
                            for model_name, model_instance in st.session_state.models.items():
                                prediction = model_instance.predict(input_features_drawn)
                                pred_ascii = int(np.round(prediction[0]))
                                predicted_char = (
                                    chr(pred_ascii) if 32 <= pred_ascii <= 126 else "?"
                                )
                                raw_output = float(prediction[0])

                                st.write(f"**{model_name} Prediction:**")
                                st.info(f"""
                                       🎯 Predicted Character: **{predicted_char}**
                                       🖥 ASCII Value: **{pred_ascii}**
                                       📊 Raw Output: **{raw_output:.2f}**
                                       """)
                            st.markdown("---")

                            st.subheader("Feature Analysis (Input to Models - Drawn Grid)")
                            st.write(f"Row sums (7 values): {row_sums}")
                            st.write(f"Column sums (5 values): {col_sums}")
                            st.write(
                                f"Combined features (12 values): {np.concatenate([row_sums, col_sums])}"
                            )
                        except Exception as e:
                            st.error(f"❌ Prediction for drawn grid failed: {str(e)}")
            
            with tab2:
                # Custom pixel input
                st.subheader("Input Custom Pixel Data")
                custom_pixel_input = st.text_input(
                    "Enter 35 pixel values (0 or 1) separated by commas (e.g., 0,1,1,1,0,1,0,...):",
                    key="custom_pixel_input",
                    help="Enter exactly 35 values for a 7x5 pixel grid"
                )
                
                # Inference button for custom input
                if st.button(
                    "🔍 Predict Character from Custom Input", 
                    disabled=not st.session_state.is_trained,
                    key="predict_custom"
                ):
                    if st.session_state.is_trained and st.session_state.models and custom_pixel_input:
                        st.subheader("Prediction Results - Custom Input")

                        # Prediction for custom input
                        try:
                            values = [int(x.strip()) for x in custom_pixel_input.split(',')]
                            if len(values) == 35: # 35 pixels only
                                pixel_values = np.array(values).reshape(7, 5)

                                # Display the custom input grid
                                st.text("Custom Input Grid (7x5):")
                                for row in pixel_values.astype(int):
                                    st.write("".join(["🟩" if p == 1 else "⬜" for p in row]))

                                row_sums_custom = np.sum(pixel_values, axis=1)
                                col_sums_custom = np.sum(pixel_values, axis=0)
                                input_features_custom = np.concatenate([row_sums_custom, col_sums_custom]).reshape(
                                    1, -1
                                )

                                for model_name, model_instance in st.session_state.models.items():
                                    prediction = model_instance.predict(input_features_custom)
                                    pred_ascii = int(np.round(prediction[0]))
                                    predicted_char = (
                                        chr(pred_ascii) if 32 <= pred_ascii <= 126 else "?"
                                    )
                                    raw_output = float(prediction[0])

                                    st.write(f"**{model_name} Prediction:**")
                                    st.info(f"""
                                           🎯 Predicted Character: **{predicted_char}**
                                           🖥 ASCII Value: **{pred_ascii}**
                                           📊 Raw Output: **{raw_output:.2f}**
                                           """)
                                st.markdown("---")

                                st.subheader("Feature Analysis (Input to Models - Custom Input)")
                                st.write(f"Row sums (7 values): {row_sums_custom}")
                                st.write(f"Column sums (5 values): {col_sums_custom}")
                                st.write(
                                    f"Combined features (12 values): {np.concatenate([row_sums_custom, col_sums_custom])}"
                                )

                            else:
                                st.warning("Please enter exactly 35 pixel values separated by commas.")
                        except ValueError:
                            st.error("Invalid input format. Please ensure all values are numbers separated by commas.")
                        except Exception as e:
                            st.error(f"❌ Prediction for custom input failed: {str(e)}")
                    elif not custom_pixel_input:
                        st.warning("Please enter custom pixel data first.")


# Run the app
if __name__ == "__main__":
    main()