from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    msg = ""

    if request.method == "POST":
        f1 = request.files["f1"]
        f2 = request.files["f2"]

        s1 = f1.read().decode("utf-8").lower()
        s2 = f2.read().decode("utf-8").lower()

        texts = [s1, s2]

        cv = CountVectorizer()
        vector = cv.fit_transform(texts)

        cs = cosine_similarity(vector)
        score = round(cs[0][1] * 100, 2)

        if score > 70:
            msg = f"⚠️ High Plagiarism Detected ({score}%)"
        elif score > 40:
            msg = f"😐 Medium Plagiarism ({score}%)"
        else:
            msg = f"✅ Negligible Plagiarism ({score}%)"

    return render_template("home.html", msg=msg)


   # app.run(debug=True)
