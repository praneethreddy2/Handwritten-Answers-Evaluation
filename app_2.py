import streamlit as st
from PIL import Image
import google.generativeai as genai
import json
import io
import re
import torch
import string
from sentence_transformers import SentenceTransformer, util


# Configure Gemini API
genai.configure(api_key="AIzaSyCLfjE99xSyFUWr5-56KoDjcHXEItrTPTY")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Streamlit UI
st.title("Context-Based Answer Sheet Evaluator")


num_pages = st.number_input("Enter number of answer sheet pages:", min_value=1, max_value=10, step=1)
uploaded_images = st.file_uploader("Upload Answer Sheet Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_json = st.file_uploader("Upload Model Answer JSON", type=["json"])


# Gemini OCR Function
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


# Extract answers from numbered OCR text
def extract_answers_in_order(ocr_text, total_questions=22):
    pattern = r"(\d{1,2})\.\s(.*?)(?=\n\d{1,2}\.\s|\Z)"
    matches = re.findall(pattern, ocr_text, flags=re.DOTALL)


    answer_dict = {str(i): "Not Answered" for i in range(1, total_questions + 1)}


    for qnum, content in matches:
        lines = content.strip().split('\n')
        answer = '\n'.join(lines[1:]).strip()
        if answer:
            answer_dict[qnum] = answer


    return [answer_dict[str(i)] for i in range(1, total_questions + 1)]


# Semantic similarity
def compute_semantic_similarity(ans1, ans2):
    emb1 = semantic_model.encode(ans1, convert_to_tensor=True,device = device)
    emb2 = semantic_model.encode(ans2, convert_to_tensor=True,device = device)
    return util.pytorch_cos_sim(emb1, emb2).item()


def normalize_text(text, remove_all_spaces=False):
    text = text.lower()

    # Replace both "acronym (full form)" and "full form (acronym)" with just acronym
    for acronym, full in acronym_map.items():
        acronym_l = acronym.lower()
        full_l = full.lower()

        # Replace full form first to prevent double match
        text = re.sub(rf"\b{re.escape(full_l)}\s*\(\s*{acronym_l}\s*\)", acronym_l, text)
        text = re.sub(rf"\b{acronym_l}\s*\(\s*{re.escape(full_l)}\s*\)", acronym_l, text)
        text = re.sub(rf"\b{re.escape(full_l)}\b", acronym_l, text)

    # Remove punctuation and normalize whitespace
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_all_spaces:
        text = text.replace(" ", "")
    
    return text

# Evaluation logic
def evaluate_answer(student_ans, qdata):
    raw_student_ans = student_ans.strip().lower()
    normalized_ans = normalize_text(raw_student_ans)
    normalized_ans_no_space = normalize_text(raw_student_ans, remove_all_spaces=True)
    max_marks = qdata.get("max_marks", 1)

    if not normalized_ans or normalized_ans in ["*", "not answered", "n/a", "na", "-"]:
        return 0

    # Range check
    if "Range" in qdata:
        for val in qdata["Range"]:
            norm_val = normalize_text(val, remove_all_spaces=True)
            if norm_val in normalized_ans_no_space:
                return max_marks
        return 0

    # Keyword checking
    if "keywords" in qdata:
        keywords = qdata["keywords"]
        crit = qdata.get("criterion", "").lower()

        if isinstance(keywords, str):
            norm_kw = normalize_text(keywords, remove_all_spaces=True)
            if norm_kw in normalized_ans_no_space:
                return max_marks
            else:
                return 0

        elif isinstance(keywords, list):
            normalized_keywords = [normalize_text(kw, remove_all_spaces=True) for kw in keywords]
            print("Normalized Student Answer (no spaces):", normalized_ans_no_space)
            print("Normalized Keywords:", normalized_keywords)
            match_count = sum(
                1 for kw in normalized_keywords
                if kw in normalized_ans_no_space
            )
        
            if "all of the above" in crit:
                return max_marks if match_count == len(keywords) else 0
            elif "atleast two" in crit:
                return max_marks if match_count >= 2 else 0
            elif "atleast one" in crit:
                return max_marks if match_count >= 1 else 0
            elif "do not give any marks if some other keywords are found" in crit:
                return max_marks if match_count == len(keywords) else 0

    # Semantic similarity
    if "model_answer" in qdata and qdata["model_answer"]:
        model_ans_norm = normalize_text(qdata["model_answer"], remove_all_spaces=True)
        sim = compute_semantic_similarity(normalized_ans_no_space, model_ans_norm)
        return max_marks if sim > 0.6 else 0

    return 0



# Main app logic
if uploaded_images and len(uploaded_images) == num_pages:
    full_text = ""
    with st.spinner("Extracting text from all pages..."):
        for img in uploaded_images:
            extracted = extract_text_from_image(img)
            full_text += extracted + "\n"


    st.subheader("Extracted Text")
    st.text_area("OCR Output", full_text, height=400)


    if uploaded_json:
        model_data = json.load(uploaded_json)
        acronym_map = model_data.get("acronym_map", {})

        st.subheader("Extracted Answers (Serial Order)")
        answers_list = extract_answers_in_order(full_text, total_questions=22)


        for idx, ans in enumerate(answers_list, start=1):
            st.markdown(f"**Q{idx}:** {ans if ans else '*Not Answered*'}")


        # Evaluation Phase
        st.subheader("Evaluation Results")
        total_score = 0
        out_of_score = 0


        #for idx, qid in enumerate(sorted(model_data.keys(), key=lambda x: int(x[1:]))):
        question_keys = sorted([k for k in model_data if k.startswith("Q")], key=lambda x: int(x[1:]))
        for idx, qid in enumerate(question_keys):
            qdata = model_data[qid]
            marks = evaluate_answer(answers_list[idx], qdata)
            total_score += marks
            out_of_score += qdata.get("max_marks", 1)


            st.markdown(f"### Q{idx + 1}: {qdata['question_text']}")
            st.markdown(f"- **Student Answer:** {answers_list[idx] if answers_list[idx] else '*Not Answered*'}")
            st.markdown(f"- **Marks Awarded:** {marks} / {qdata['max_marks']}")


        st.success(f"**Total Score: {total_score} / {out_of_score}**")