# ⚖️ LexBanglaRAG: Evidence-Grounded Statutory Legal AI Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Corpus](https://img.shields.io/badge/Corpus-49%20Acts%20%7C%201%2C575%20Sections-green.svg)](https://github.com/)
[![Benchmark](https://img.shields.io/badge/Benchmark-180%20Gold%20QA-orange.svg)](https://github.com/)
[![Hardware](https://img.shields.io/badge/Hardware-16GB%20T4%20GPU-purple.svg)](https://github.com/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An evidence-grounded Retrieval-Augmented Generation (RAG) interactive web application and benchmarking system for **Bengali Statutory Question Answering and Hallucination Analysis**. 

Developed at the **Department of Software Engineering, Daffodil International University (DIU)**.

---

## 📌 System Architecture

![System Architecture](figures/fig1_system_architecture.png)

---

## ✨ Key Features of the Web Interface

1. **🔍 Interactive Statutory Question Answering:** 
   - Ask any legal dispute question in colloquial Bengali.
   - Or select from **10+ real-world citizen dispute archetypes** across major Bangladeshi Acts (Labor Act, EPZ Labor, Child Rights, Road Transport, Consumer Rights, Dowry Prohibition, Cyber Security, etc.).
2. **📚 Transparent Evidence Inspection (Retriever):**
   - View the exact retrieved parliamentary Act, section number, chapter metadata, and verbatim legal text.
3. **⚡ RAG vs. No-RAG Hallucination Comparison (Supervisor Highlight):**
   - Compare side-by-side:
     - **LexBanglaRAG (Evidence-Grounded):** Cites verified Act titles and section numbers with legal grounding.
     - **Closed-Book AI (No-RAG Baseline):** Demonstrates parametric hallucination, showing how standard LLMs fabricate nonexistent sections (73.89% pure fabrication rate).
4. **📜 49 Parliamentary Acts Corpus Explorer:**
   - Instant search and exploration of all 1,575 codified statutory sections from 1985 to 2025.
5. **📊 Benchmark Metrics Dashboard:**
   - Full empirical evaluation tables across Qwen2.5 (3B, 7B, 14B) and Gemma-2 (2B).

---

## 🚀 How to Run Locally

### Option 1: One-Click Launcher (Windows)
Simply double-click the **`run_demo.bat`** file in the project folder. It will automatically check dependencies and launch the app in your browser!

### Option 2: Via Terminal / PowerShell
1. Open PowerShell or Command Prompt in this folder.
2. Install the lightweight requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. The application will open automatically in your browser at:
   `http://localhost:8501`

---

## 🌐 How to Deploy for Free on Hugging Face Spaces (Live Link for Supervisor)

1. Create a free account at [Hugging Face](https://huggingface.co/).
2. Click **New Space** (https://huggingface.co/new-space).
3. Fill in:
   - **Space Name:** `LexBanglaRAG`
   - **SDK:** `Streamlit`
   - **License:** `Open Source (MIT / CC-BY-4.0)`
4. Upload all files from this folder (`app.py`, `requirements.txt`, `README.md`, `data/`, `figures/`).
5. Hugging Face will automatically build and give you a public URL (e.g., `https://huggingface.co/spaces/your-username/LexBanglaRAG`) that your supervisor can open directly on their phone or laptop!

---

## 📊 Summary of Research Findings

| Model Configuration | Precision | Act Citation Accuracy | Section Citation Accuracy | Token F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B No-RAG (Closed-Book)** | FP16 | 13.89% | 18.33% (Strict: 0%) | 0.1870 |
| **Qwen2.5-14B No-RAG (Closed-Book)**| 4-bit NF4 | 16.67% | 14.44% (Strict: 1.67%) | 0.1980 |
| **Gemma-2-2B + LexBanglaRAG** | FP16 | 43.89% | 46.11% | 0.2974 |
| **Qwen2.5-3B + LexBanglaRAG** | FP16 | 50.56% | 50.00% | 0.3370 |
| **Qwen2.5-14B + LexBanglaRAG (Peak)**| 4-bit NF4 | **55.56%** | **53.89%** | **0.3528** |

- **Statutory Context Enrichment** boosts dense retrieval MRR to **0.8250** (Recall@20: 98.89%).
- Evidence grounding slashes pure statutory fabrication by **29.45 percentage points** ($p < 10^{-12}$).

---

## 🎓 Academic Affiliation
* **Department:** Department of Software Engineering (Major in Data Science)
* **Institution:** Daffodil International University (DIU), Dhaka, Bangladesh
* **Year:** 2026
