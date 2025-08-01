import streamlit as st
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from PIL import Image
import google.generativeai as genai
import json
import io
import re
import openai
from openai import OpenAI
import anthropic
import time

# Configure API keys
genai.configure(api_key= "AIzaSyCLfjE99xSyFUWr5-56KoDjcHXEItrTPTY")
openai_client = openai.OpenAI(api_key = "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")
anthropic_client = anthropic.Anthropic(api_key="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")

# Device setup

# Streamlit UI
st.title("Context-Based Answer Sheet Evaluator (LLM Majority Voting)")

num_pages = st.number_input("Enter number of answer sheet pages:", min_value=1, max_value=10, step=1)
uploaded_images = st.file_uploader("Upload Answer Sheet Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_json = st.file_uploader("Upload Model Answer JSON", type=["json"])

# Gemini OCR function
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

# Extract answers based on numbering
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

# --- LLM Evaluation Functions ---

# OpenAI evaluation
def evaluate_with_openai(question, student_answer, max_marks):
    time.sleep(70)
    print("\n Question given to OPenai : ",question)
    print("\n student answer given to Openai : ",student_answer)
    print("\n maximum marks given to OpenAI : ",max_marks)
    prompt = f"""
You are an expert academic evaluator.

Question: "{question}"

Student's Answer: "{student_answer}"

Maximum Marks: {max_marks}

Evaluate following these instructions:
- Full marks for fully correct answers.
- Give full marks even if phrased differently.
- 0 marks for completely wrong answers.
- Ignore minor spelling or language mistakes.

Return ONLY a single number between 0 and {max_marks}.
"""
    print("\n Prompt given to OpenAi model : ",prompt)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert academic evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        print("\n Response given by OpenAi : ",response)
        mark = int(float(response['choices'][0]['message']['content'].strip()))
        print("\n marks awarded by OpenAi : ",mark)
        return min(max(mark, 0), max_marks)
    except Exception as e:
        print(f"[Openai Error] {e}")
        return 0

# Anthropic (Claude) evaluation
def evaluate_with_claude(question, student_answer, max_marks):
    time.sleep(60)
    print("\n Question: ",question)
    print("\n student answer : ",student_answer)
    print("\n maximum marks : ",max_marks)
    prompt = f"""
You are an expert academic evaluator.

Question: "{question}"
Student's Answer: "{student_answer}"
Maximum Marks: {max_marks}

Evaluate following these instructions:
- Full marks for fully correct answers.
- Give full marks even if phrased differently.
- 0 marks for completely wrong answers.
- Ignore minor spelling or language mistakes.

Return ONLY a single number between 0 and {max_marks}.
"""
    print("\n Prompt sent to Anthropic Claude : ",prompt)

    try:
        response = anthropic_client.messages.create(
            model="Claude 3 Haiku",
            max_tokens=10,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print("\n resonse from Anthropic Claude : ",response)
        content = response.content[0].text.strip()
        mark = int(float(content))
        print("\n marks awarded by Anthropic Claude : ",mark)
        return min(max(mark, 0), max_marks)
    except Exception as e:
        print(f"[Claude Error] {e}")
        return 0

# Gemini evaluation
def evaluate_with_gemini(question, student_answer, max_marks):
    time.sleep(4)
    print("\n Question given to gemini model : ",question)
    print("\n student answer given to gemini model : ",student_answer)
    print("\n maximum marks given to Gemini model : ",max_marks)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are an expert academic evaluator.

Question: "{question}"
Student's Answer: "{student_answer}"
Maximum Marks: {max_marks}

Evaluate following these instructions:
- Full marks for fully correct answers.
- Give full marks even if phrased differently.
- 0 marks for completely wrong answers.
- Ignore minor spelling or language mistakes.

Return ONLY a single number between 0 and {max_marks} and also give reason why you have awarded this mark.
"""
    print("\n Prompt given to Gemini model : ",prompt)

    try:
        response = model.generate_content(prompt)
        match = re.search(r'\d+(\.\d+)?', response.text)
        mark = int(float(match.group())) if match else 0
        #mark = int(float(response.text.strip()))
        print("\n Response given by Gemini model : ",response)
        print("\n marks awarded by Gemini model : ",mark)
        return min(max(mark, 0), max_marks)
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return 0

# --- Majority Voting Function ---

def majority_vote(marks_list):
    return int(round(sum(marks_list) / len(marks_list)))

# --- Main App Logic ---

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

        st.subheader("Extracted Answers (Serial Order)")
        answers_list = extract_answers_in_order(full_text, total_questions=22)

        for idx, ans in enumerate(answers_list, start=1):
            st.markdown(f"**Q{idx}:** {ans if ans else '*Not Answered*'}")

        # --- Evaluation Phase ---
        st.subheader("Evaluation Results")
        total_score = 0
        out_of_score = 0

        for idx, (qid, qdata) in enumerate(sorted(model_data.items(), key=lambda x: int(re.sub(r'\D', '', x[0])))):
            question_text = qdata.get("question_text", "")
            max_marks = qdata.get("max_marks", 1)
            student_answer = answers_list[idx]

            if student_answer.lower() in ["not answered", "", "n/a", "na"]:
                final_mark = 0
            else:
                #openai_mark = evaluate_with_openai(question_text, student_answer, max_marks)
                #print("Response ")
                #claude_mark = evaluate_with_claude(question_text, student_answer, max_marks)
                gemini_mark = evaluate_with_gemini(question_text, student_answer, max_marks)

                #all_marks = [openai_mark, claude_mark, gemini_mark]
                #final_mark = majority_vote(all_marks)
                final_mark = gemini_mark

            total_score += final_mark
            out_of_score += max_marks

            st.markdown(f"### Q{idx + 1}: {question_text}")
            st.markdown(f"- **Student Answer:** {student_answer if student_answer else '*Not Answered*'}")
            st.markdown(f"- **Marks Awarded (Final after voting):** {final_mark} / {max_marks}")

        st.success(f"**Total Score: {total_score} / {out_of_score}**")
