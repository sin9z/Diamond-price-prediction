import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

__version__ = "1.0.0"

# Load model
model = load_model("C:/Users/Hello/OneDrive/Desktop/ds projects/diamond price project/tf_m_1.0.0.h5", compile=False)
model.compile(loss='mean_squared_error', optimizer='adam')

features = [
    'cut', 'color', 'clarity', 'carat_weight', 'cut_quality', 'lab', 
    'symmetry', 'polish', 'eye_clean', 'culet_size', 'culet_condition', 
    'depth_percent', 'table_percent', 'meas_length', 'meas_width', 
    'meas_depth', 'girdle_min', 'girdle_max', 'fluor_color', 
    'fluor_intensity', 'fancy_color_dominant_color', 'fancy_color_secondary_color', 
    'fancy_color_overtone', 'fancy_color_intensity'
]

def predict_pipe(input_list):
    try:
        # Convert input list to DataFrame
        input_df = pd.DataFrame([input_list], columns=features)
        
        # Process categorical features with encoders
        for i, feature in enumerate(features):
            if isinstance(input_list[i], str):  # Check if categorical
                encoder_path = f"C:/Projects/Diamond Sales Prediction/{feature}.joblib"
                try:
                    encoder = joblib.load(encoder_path)
                    input_df[feature] = encoder.transform([input_list[i]])[0]
                except FileNotFoundError:
                    raise ValueError(f"Encoder for feature '{feature}' not found at {encoder_path}")
            else:
                input_df[feature] = float(input_list[i])  # Ensure numeric

        # Convert to NumPy array for model prediction
        xtest = np.array(input_df.astype(np.float64))
        prediction = model.predict(xtest)
        return prediction[0][0]
    except Exception as e:
        raise ValueError(f"Error in prediction pipeline: {e}")
