# AI Resume Screening Assistant

A recruiter's screening tool built with **LangChain** and **RAG (Retrieval-Augmented Generation)**.
Upload one or more resumes (PDF), paste a job description, and get a grounded, structured
evaluation of each candidate — match score, matching and missing skills, strengths,
weaknesses, and a hiring recommendation.

The assistant answers **only** from the uploaded resumes. Nothing is invented.

**Live app:** _add your Streamlit Cloud link here_

---

## What it does

| Tab | What it's for |
|---|---|
| **Individual Evaluations** | Full evaluation of each resume: score gauge, verdict stamp, skills, strengths, weaknesses, recommendation |
| **Compare Two** | Put two candidates side by side and get a verdict on the stronger fit |
| **Rank & Recommend** | Rank every candidate by score and generate a final hiring recommendation |
| **Ask a Resume** | Ask free-form questions about a single resume (RAG Q&A), with the retrieved context shown |

---

## How it works

```
Resume PDF
    │
    ├─ PyPDFLoader                    load the document
    ├─ RecursiveCharacterTextSplitter chunk it (1000 / 200)
    ├─ OpenAIEmbeddings               embed the chunks
    └─ FAISS                          vector store (one per resume)
                                          │
Job Description ──────► retriever (k=5) ──┘
                                          │
                            ChatPromptTemplate
                                          │
                            ChatOpenAI (gpt-4o-mini)
                                          │
                            PydanticOutputParser
                                          │
                            structured evaluation
```

The chain is the standard LCEL pattern: `prompt | llm | parser`.

**Stack**

- LangChain (`langchain-core`, `langchain-community`, `langchain-openai`)
- LLM: OpenAI `gpt-4o-mini` (switchable in the sidebar)
- Embeddings: `text-embedding-3-small`
- Vector store: FAISS (in-memory, one per resume)
- Output parser: `PydanticOutputParser` for structured responses
- UI: Streamlit

---

## Run it locally

```bash
# 1. clone and enter the project
git clone <your-repo-url>
cd resume_screener

# 2. create an environment and install
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS / Linux

pip install -r requirements.txt

# 3. add your OpenAI key
#    create a file named .env containing:
#    OPENAI_API_KEY=sk-your-key-here

# 4. run
streamlit run app.py
```

You can also skip the `.env` file and paste your key straight into the
**API Key** field in the sidebar.

---

## Deploy on Streamlit Cloud

1. Push the repo to GitHub (confirm `.env` is **not** committed — it's in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, select the repo.
3. Set the main file to `app.py` and deploy.
4. Leave **Secrets** empty — each visitor enters their own OpenAI key in the sidebar.

> If you'd rather the app run on your own key, add
> `OPENAI_API_KEY = "sk-..."` under **Settings → Secrets**. Note that every
> evaluation any visitor runs will then be billed to your account.

---

## Test cases

| # | Scenario | Expected outcome |
|---|---|---|
| 1 | Evaluate a strong Data Scientist resume | High match score, verdict **Hire** |
| 2 | Compare two resumes against the same JD | Side-by-side scores + a reasoned verdict |
| 3 | Screen a weak/adjacent resume | Missing skills clearly listed, verdict **Pass** |
| 4 | Upload several resumes | Ranked shortlist + best-candidate recommendation |
| 5 | Ask a question about one resume | Answer grounded in that resume only |

---

## Project structure

```
resume_screener/
├── .streamlit/
│   └── config.toml       theme
├── app.py                Streamlit UI, schema, prompts, chains
├── rag_build.py          RAG build pipeline (load → chunk → embed → FAISS)
├── requirements.txt
├── .gitignore
└── .env                  your API key (never committed)
```
