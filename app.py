from flask import Flask, render_template, request, jsonify
import os
from training.train import train_selected_model, predict_move, ACTIVE_MODELS

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train-model", methods=["POST"])
def train_model():
    data = request.json
    algorithm = data.get("algorithm", "")
    hyperparams = data.get("hyperparams", {})
    if not algorithm:
        return jsonify({"message": "No algorithm specified"}), 400
    result = train_selected_model(algorithm, hyperparams)
    chart_b64 = result.get("chart_b64", None)
    return jsonify({
        "message": "Training complete!",
        "details": result,
        "chart_b64": chart_b64
    })

@app.route("/list-models", methods=["GET"])
def list_models():
    path = "saved_models"
    if not os.path.exists(path):
        return jsonify({"files": []})
    files = [f for f in os.listdir(path) if f.endswith((".pth", ".npz"))]
    return jsonify({"files": files})

@app.route("/select-model", methods=["POST"])
def select_model():
    data = request.json
    algorithm = data.get("algorithm", "")
    filename = data.get("filename", "")

    if filename:
        full_path = os.path.join("saved_models", filename)
        if not os.path.isfile(full_path):
            return jsonify({"message": f"File '{filename}' not found."}), 400

        # Very simple logic: if the filename contains "_reinforce_", we assume "reinforce"
        # else if it contains "_random-search_", we assume "random-search"
        if "_reinforce_" in filename:
            real_algorithm = "reinforce"
            ACTIVE_MODELS[real_algorithm] = {
                "algorithm": real_algorithm,
                "model_path": full_path
            }
        elif "_random-search_" in filename:
            real_algorithm = "random-search"
            ACTIVE_MODELS[real_algorithm] = {
                "algorithm": real_algorithm,
                "model_path": full_path
            }
        elif "_ars_" in filename:
            real_algorithm = "ars"
            ACTIVE_MODELS[real_algorithm] = {
            "algorithm": real_algorithm,
            "model_path": full_path
        }     
        else:
            # Fallback guess
            real_algorithm = "reinforce"
            ACTIVE_MODELS[real_algorithm] = {
                "algorithm": real_algorithm,
                "model_path": full_path
            }

        return jsonify({
            "message": f"Loaded '{filename}' as {real_algorithm} model for play.",
            "algorithm": real_algorithm
        })

    if algorithm:
        # Selecting an in-memory model by 'algorithm'
        if algorithm in ACTIVE_MODELS:
            return jsonify({"message": f"Model '{algorithm}' is selected and ready to play!", 
                            "algorithm": algorithm})
        return jsonify({"message": f"No model found in memory for '{algorithm}'."}), 400

    return jsonify({"message": "No algorithm or filename provided."}), 400

@app.route("/model-move", methods=["POST"])
def model_move():
    data = request.json
    algorithm = data.get("algorithm", "")
    board_state = data.get("board", [])
    if not algorithm:
        return jsonify({"error": "No algorithm specified"}), 400
    move = predict_move(algorithm, board_state)
    if move is None:
        return jsonify({"error": "No valid move found or no model loaded."}), 400
    return jsonify({"move": move})

if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    app.run(debug=True)
