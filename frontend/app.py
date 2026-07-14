import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, request
from rag.rag_pipeline import rag_answer
from rag.rag_pipeline import rag_cv_match

import markdown
import fitz


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

    answer = None
    sources = []
    query = ""
    result = None
    text = ""


    if request.method == "POST":

        mode = request.form.get("mode")


        # обычный поиск
        if mode == "text":

            query = request.form.get("query", "")

            if query:

                result = rag_answer(
                    query,
                    pretty_print=False
                )


        elif mode == "cv":
        
            cv = request.files.get("cv")
        
            if cv:
                text = read_pdf(cv)
        
                result = rag_cv_match(
                    text,
                    pretty_print=False)

        if result:

            answer = markdown.markdown(
                result["answer"],
                extensions=[
                    "tables"
                ]
            )

            sources = result.get(
                "sources",
                []
            )



    return render_template(
        "index.html",
        query=query,
        answer=answer,
        sources=sources
    )


def read_pdf(pdf_file):
    text = ""

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in doc:
        text += page.get_text()

    doc.close()

    return text


if __name__ == "__main__":
    app.run(debug=False)