import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, request
from rag.rag_pipeline import rag_answer

import markdown


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

    answer = None
    sources = []
    query = ""
    result = None


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


        # CV поиск пока отключен
        # elif mode == "cv":
        #
        #     cv = request.files.get("cv")
        #
        #     if cv:
        #         text = cv.read().decode(
        #             "utf-8",
        #             errors="ignore"
        #         )
        #
        #         result = cv_recommend(text)



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


if __name__ == "__main__":
    app.run(debug=False)