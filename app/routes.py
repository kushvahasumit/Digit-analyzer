import numpy as np
from flask import render_template, request, jsonify
from app import app
from app.predictors import RegressionPredictor, CNNPredictor

try:
    regression_predictor = RegressionPredictor()
    cnn_predictor = CNNPredictor()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    regression_predictor = None
    cnn_predictor = None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        print("\n===========================")
        print("📥 RAW INPUT RECEIVED")
        print("Length:", len(data))

        arr = np.array(data).astype("float32")
        print("📌 ARRAY SHAPE:", arr.shape)

        # ---------- DEBUG BEFORE RESHAPE ----------
        if arr.size != 784:
            print("❌ ERROR: INPUT SIZE IS NOT 784!")
        # ------------------------------------------

        print("\n🚀 Calling RegressionPredictor...")
        reg_pred = RegressionPredictor.predict(arr)

        print("🚀 Calling CNNPredictor...")
        cnn_pred = CNNPredictor.predict(arr)

        print("\n📤 PREDICTION SUCCESS")
        print("Regression:", reg_pred)
        print("CNN:", cnn_pred)

        print("===========================\n")

        return jsonify({"status": "ok", "data": [reg_pred, cnn_pred]})

    except Exception as e:
        import traceback
        print("\n❌❌ FULL ERROR BELOW ❌❌")
        traceback.print_exc()
        print("❌ ERROR MESSAGE:", str(e))

        return jsonify({"status": "error", "message": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500