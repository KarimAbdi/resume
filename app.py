from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load GPT-2 model for text generation
generator = pipeline("text-generation", model="gpt2")

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    resume = data.get("resume", "")
    job_desc = data.get("jobDescription", "")

    if not resume or not job_desc:
        return jsonify({"error": "Missing resume or job description"}), 400

    # Pull out keywords manually from job description (simple method)
    keywords = []
    for word in job_desc.split():
        word_clean = word.strip(".,:;()[]{}").lower()
        if len(word_clean) > 4 and word_clean not in keywords:
            keywords.append(word_clean)
    keywords = keywords[:10]  # limit to top 10

    prompt = f"""
You are an AI resume expert. Improve the resume below by:
- Rewriting sentences clearly and professionally
- Emphasizing these keywords: {', '.join(keywords)}
- Using bullet points where possible
- Making the tone confident and suitable for the job

Job Description:
{job_desc}

Original Resume:
{resume}

Optimized Resume:
"""

    try:
        result = generator(prompt, max_length=300, do_sample=True, temperature=0.7)[0]['generated_text']
        return jsonify({"optimized": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)