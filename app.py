from flask import Flask
import json
from flask import  jsonify, request
from Question_Answering import Answering 
from Search_Pinecone import SearchSimilarity


app = Flask(__name__)

@app.route("/answer", methods=["POST"])
def answer():
    body = request.get_json()

    if "text" not in body :
        error = {
            "code": 400,
            "message": "REQUIRE FIELD 'text'"
        }
        return jsonify(error), 400
    text = body["text"]
    document , question =  SearchSimilarity(text)
    result = Answering(document,question)
    return json.dumps({
        "status": True,
        "code": 200,
        "message": "Success",
        "results": result,
        "reference": document
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8400)