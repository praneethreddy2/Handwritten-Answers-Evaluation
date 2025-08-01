#  Master's Thesis Project: Intelligent Answer Sheet Evaluation (MTP-1 & MTP-2)

##  Overview

This repository documents the work carried out as part of my master’s thesis in two phases:

- **MTP-1**: Combined research and development of automated evaluation systems for short and descriptive answers.
- **MTP-2**: Current phase focused on application-based and analytical question evaluation using LangChain and LLMs.

---

##  MTP-1: Research + Development

###  Phase 1 - Research on Handwritten Answer Sheets

Initially, the goal was to evaluate **handwritten student responses** collected through paper-based exams. The challenges led to a detailed investigation of:

- OCR tools tested: `Tesseract`, `EasyOCR`, `PaddleOCR`, `TrOCR`
- Custom bounding box detection using MMDetection and RCNN-based models
- Issues encountered:
  - Curved lines, non-uniform handwriting styles
  - OCR accuracy limited (~50–60%)
  - Poor generalization across answer sheets

> **Conclusion**: Due to unreliable OCR on real-world handwritten scripts, the project pivoted to structured digital input and robust evaluation logic.

---

###  Phase 2 - Application Development

To create a practical and scalable solution, we developed **three types of evaluation applications**:

#### 1️⃣ Model-Answer Based Evaluation App

- Accepts student answers via Google Forms
- Compares answers against a JSON-configured model answer bank
- Evaluation logic includes:
  - **Keyword matching**
  - **Numeric range checking**
  - **Regex-based pattern matching**
- Fully **explainable and customizable** rules per question

#### 2️⃣ Gemini-based LLM Evaluation App

- Uses Google’s **Gemini LLM API** to evaluate each student answer
- Dynamically generated prompts per question
- Allows **context-based reasoning** beyond rigid keyword matching

#### 3️⃣ Hybrid Evaluation App

- Integrates rule-based and LLM-based scoring
- Falls back to Gemini for cases where keyword-based rules fail
- Offers **best of both worlds**: explainability + adaptability

###  Key Techniques

- **Prompt Engineering**: Inject constraints into Gemini prompts for controlled scoring
- **Regular Expressions**: Extract patterns, acronyms, and validate structured answers
- **Rule Templates**: Custom evaluation instructions per question from JSON config

> ⚠️ **Note**: The answer sheets and form responses used in this project cannot be publicly shared due to academic privacy constraints.

---

##  MTP-2: LangChain-Based Intelligent QA (Ongoing)

###  Objective

- Develop real-time systems to evaluate **application-level** and **analysis-based** answers
- Replace traditional evaluation rubrics with LLM-backed reasoning agents

##  Screenshots

###   Fine tuning approach
![Sentence Database](<img width="2479" height="3542" alt="a01-000u" src="https://github.com/user-attachments/assets/f849089f-ab15-4d51-b69c-d138d5b2940a" />
)
<img width="637" height="362" alt="image" src="https://github.com/user-attachments/assets/bac43b17-9fac-49ec-a626-73872808c73b" />


### Dataset used for evaluation
![Dataset](<img width="916" height="535" alt="image" src="https://github.com/user-attachments/assets/6aa88dbd-2036-453d-8c3e-aff862273745" />
)

### Interface 
<img width="627" height="372" alt="image" src="https://github.com/user-attachments/assets/833ba88c-9ad1-4dea-b109-cbcfda6f1839" />

## Model answer
<img width="927" height="197" alt="image" src="https://github.com/user-attachments/assets/9742faed-dda4-430e-a8e7-8281bb085c35" />

## Extracted text
<img width="618" height="422" alt="image" src="https://github.com/user-attachments/assets/e25a4dbf-bd5f-4e21-b943-cd9627444701" />
<img width="438" height="412" alt="image" src="https://github.com/user-attachments/assets/64182ee4-3ae7-4af7-b634-2f9cc644ca32" />

## Graphical Analysis
<img width="812" height="396" alt="image" src="https://github.com/user-attachments/assets/6d8697b8-1e67-4a95-8aec-222f06a431ae" />









