from flask import Flask, request, jsonify

# importing subprocess module
import subprocess

# running other file using run()
app = Flask(__name__)


# Route to display the output of product-recommendation.py
@app.route('/get_recommendations',  methods=['GET'])
def get_recommendations():
    subprocess.run(["python", "product-recommendation.py"])
    return "Done!"

# Route to take user input and pass it to product-recommendation.py

if __name__ == "__main__":
    app.run(debug=True, port=8000)