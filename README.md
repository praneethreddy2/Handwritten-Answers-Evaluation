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





