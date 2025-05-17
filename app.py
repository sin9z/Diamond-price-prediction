from flask import Flask, render_template, request
import model as m

app = Flask(__name__)

features = [
    'cut', 'color', 'clarity', 'carat_weight', 'cut_quality', 'lab', 
    'symmetry', 'polish', 'eye_clean', 'culet_size', 'culet_condition', 
    'depth_percent', 'table_percent', 'meas_length', 'meas_width', 
    'meas_depth', 'girdle_min', 'girdle_max', 'fluor_color', 
    'fluor_intensity', 'fancy_color_dominant_color', 'fancy_color_secondary_color', 
    'fancy_color_overtone', 'fancy_color_intensity'
]

@app.route("/", methods=['GET', 'POST'])
def home():
    sales_prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            # Gather input values from form
            input_list = []
            for feature in features:
                value = request.form.get(feature, '')
                try:
                    input_list.append(float(value))  # Convert numeric inputs
                except ValueError:
                    input_list.append(value)  # Keep string inputs

            # Make prediction using model
            sales_prediction = m.predict_pipe(input_list)
        except Exception as e:
            error_message = str(e)

    return render_template('index.html', sale=sales_prediction, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
