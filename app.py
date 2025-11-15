from flask import Flask, request, jsonify, render_template
from tokenizer import run_tokenizer

app = Flask(__name__)

@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    data = request.get_json() or {}
    words = data.get("words", [])
    results = run_tokenizer(words)

    out = []
    for r in results:
        out.append({
            "token": r["word"],
            "status": r["status"],
            "first_token": r["first_split"]["left"],
            "second_token": r["first_split"]["right"],
        })
    return jsonify(out)

@app.route("/")
def index():
    return render_template("index.html")
