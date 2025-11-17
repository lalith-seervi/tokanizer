from flask import Flask, render_template, request, jsonify
from tokenizer import run_tokenizer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    data = request.get_json() or {}
    words = data.get("words", [])
    results = run_tokenizer(words)

    out = []
    for r in results:
        if(r["status"] == "Valid split"):
            out.append({
                "token": r["word"],
                "status": r["status"],
                "first_token": r["best_split"]["left"],
                "second_token": r["best_split"]["right"],
            })
        else:
            out.append({
                "token": r["word"],
                "status": r["status"],
            })
    return jsonify(out)
