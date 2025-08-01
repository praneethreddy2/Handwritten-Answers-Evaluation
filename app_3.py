import streamlit as st
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from PIL import Image
import google.generativeai as genai
import json
import io
import re
import time

# Configure Gemini API
genai.configure(api_key="AIzaSyCLfjE99xSyFUWr5-56KoDjcHXEItrTPTY")  # Replace with your actual key

# Streamlit App Title
st.title("Context-Based Answer Sheet Evaluator (Gemini Model)")

# User inputs
num_pages = st.number_input("Enter number of answer sheet pages:", min_value=1, max_value=10, step=1)
uploaded_images = st.file_uploader("Upload Answer Sheet Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_json = st.file_uploader("Upload Model Answer JSON", type=["json"])

# OCR with Gemini
def extract_text_from_image(image_file):
    model = genai.GenerativeModel("gemini-1.5-flash")
    img = Image.open(image_file).convert("RGB")
    byte_io = io.BytesIO()
    img.save(byte_io, format="PNG")
    byte_data = byte_io.getvalue()

    response = model.generate_content([
        "Extract all visible questions and handwritten answers from this image clearly and in sequence. Maintain the original format.",
        {"mime_type": "image/png", "data": byte_data}
    ])
    return response.text

# Extract answers in numerical order
def extract_answers_in_order(ocr_text, total_questions):
    pattern = r"(\d{1,2})\.\s(.*?)(?=\n\d{1,2}\.\s|\Z)"
    matches = re.findall(pattern, ocr_text, flags=re.DOTALL)
    answers = {str(i): "Not Answered" for i in range(1, total_questions + 1)}
    for qnum, content in matches:
        lines = content.strip().split("\n")
        answer = "\n".join(lines[1:]).strip()
        if answer:
            answers[qnum] = answer
    return [answers[str(i)] for i in range(1, total_questions + 1)]


# Gemini Evaluation
import time
import re
import google.generativeai as genai

def evaluate_with_gemini(question, student_answer, max_marks,
                         criterion=None, keywords=None,
                         value_range=None, model_answer=None):
    time.sleep(4)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Construct evaluation instructions
    if model_answer:
        criterion_text = (
            "Do not rely on keyword matching or require exact phrasing. "
            "Evaluate based solely on whether the student’s answer demonstrates conceptual understanding "
            "and correctly conveys the intended meaning of the Reference Answer.\n\n"
            "If the student's answer presents the correct components, steps, or explanation (even broken into parts), "
            "and the overall idea is consistent with the reference, award appropriate marks. "
            "Reasonable variations in wording, examples, or order of explanation should not be penalized. "
            "If a numerical range is inferred by summing logical components, count that as valid."
        )
    elif keywords:
        criterion_text = (
            "Award full marks if the student mentions the correct items from the keyword list. "
            "They should not include incorrect items that are not in the list. "
            "Do not penalize for omitting incorrect options that were not in the list. "
            "Award marks only if the answer reflects understanding of the correct options."
        )
    else:
        criterion_text = criterion or (
            "Award full marks if the student answer aligns in meaning and concept with the Reference Answer; otherwise, award 0."
        )

    # Add any additional context
    extra = ""
    if keywords:
        extra += f"- Keywords (correct options): {', '.join(keywords)}\n"
    if value_range:
        extra += f"- Valid Range: {', '.join(value_range)}\n"
    if model_answer:
        extra += f"- Reference Answer: {model_answer}\n"

    # Build full prompt
    prompt = f"""
You are an expert academic evaluator.

Your task is to assess a student's handwritten answer using ONLY the information provided below.

Question:
{question}

Student's Answer:
{student_answer}

Maximum Marks: {max_marks}

Evaluation Criterion:
{criterion_text}

Additional Context:
{extra}

### Instructions:
- Focus on meaning and understanding, not wording or formatting.
- Do NOT require exact matches with keywords like "yes" unless the meaning is significantly affected.
- Award marks if the student has correctly addressed the necessary ideas or components — even if they explain them separately or in a different order.
- Be fair in recognizing partial correctness if the student's answer demonstrates partial understanding.

### Output Format:
Reason: A brief explanation of how the student answer compares conceptually to the reference.
"""

    try:
        resp = model.generate_content(prompt)
        txt = resp.text.strip()
        m = re.search(r"\b(\d+)\b", txt)
        mark = int(m.group(1)) if m else 0
        reason = txt[m.end():].strip() if m else txt
        return min(max(mark, 0), max_marks), reason
    except Exception as e:
        print("[Gemini Error]", e)
        return 0, "Error during evaluation."



# Main App Logic
if uploaded_images and uploaded_json and len(uploaded_images) == num_pages:
    full_text = ""
    with st.spinner("Extracting text..."):
        for img in uploaded_images:
            full_text += extract_text_from_image(img) + "\n"
    st.subheader("OCR Output")
    st.text_area("", full_text, height=300)

    try:
        model_data = json.load(uploaded_json)
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        st.stop()

    total_questions = len([k for k in model_data if re.match(r'^Q\d+$', k)])
    answers = extract_answers_in_order(full_text, total_questions)

    st.subheader("Extracted Answers")
    for i, ans in enumerate(answers, start=1):
        st.markdown(f"**Q{i}:** {ans}")

    # Evaluation
    st.subheader("Evaluation Results")
    total, out_of = 0, 0
    qkeys = sorted([k for k in model_data if re.match(r'^Q\d+$', k)], key=lambda x: int(x[1:]))

    for idx, qid in enumerate(qkeys, start=1):
        qd = model_data[qid]
        question = qd.get('question_text', '')
        max_m = qd.get('max_marks', 1)
        criterion = qd.get('criterion', None)
        keywords = qd.get('keywords', None)
        rng = qd.get('Range', None)
        m_ans = qd.get('model_answer', None)
        s_ans = answers[idx - 1]

        if not s_ans or s_ans.lower() in ['not answered', 'n/a', 'na']:
            mark, reason = 0, "Not answered"
        else:
            mark, reason = evaluate_with_gemini(
                question, s_ans, max_m,
                criterion=criterion,
                keywords=keywords,
                value_range=rng,
                model_answer=m_ans
            )

        total += mark
        out_of += max_m

        st.markdown(f"### Q{idx}: {question}")
        st.markdown(f"- **Answer:** {s_ans}")
        st.markdown(f"- **Marks:** {mark} / {max_m}")
        st.markdown(f"- **Reason:** {reason}")

    st.success(f"**Total Score: {total} / {out_of}**")
