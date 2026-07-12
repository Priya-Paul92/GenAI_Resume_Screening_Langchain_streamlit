#  app.py
#  AI Resume Screening Assistant  —  LangChain + RAG (FAISS)
#  Upload resumes (PDF), paste a Job Description, and get grounded, structured
#  candidate evaluations. The LLM answers only from the uploaded resumes.

import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from rag_build import build_vectorstore, EMBEDDING_MODEL

load_dotenv()

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===============================================================
#  Styling
# ===============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@700&display=swap');

.stApp {
    background:
        radial-gradient(1100px 600px at 8% 0%, #DCEFE9 0%, rgba(220,239,233,0) 55%),
        radial-gradient(900px 600px at 95% 8%, #E7E2F2 0%, rgba(231,226,242,0) 55%),
        linear-gradient(160deg, #F5F2EB 0%, #EFF3F1 55%, #F3EFE8 100%);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: #FFFFFF;
    border: 1px solid #E2E0D8;
    border-left: 5px solid #0E7C6B;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 6px 22px rgba(18,48,58,0.06);
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px; font-weight: 700; color: #12303A;
    margin: 0 0 6px 0; letter-spacing: -0.5px;
}
.hero p { color: #55676E; font-size: 14px; margin: 0; }

.cand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #12303A; margin: 0;
}
.cand-file { color: #8A9AA1; font-size: 12px; margin: 2px 0 0 0; }

/* verdict stamp */
.stamp {
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700; font-size: 13px;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 6px 14px; border-radius: 4px;
    border: 2px solid currentColor;
    transform: rotate(-2deg);
    opacity: 0.92;
}
.stamp-hire  { color: #0E7C6B; background: #E4F2EE; }
.stamp-maybe { color: #B07908; background: #FBF1DC; }
.stamp-pass  { color: #C2410C; background: #FBE7DC; }

.sec-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: 1.6px;
    text-transform: uppercase; color: #8A9AA1;
    margin: 14px 0 8px 0;
}

.chip {
    display: inline-block; padding: 5px 11px; margin: 0 6px 6px 0;
    border-radius: 999px; font-size: 12.5px; font-weight: 500; line-height: 1.5;
}
.chip-match { background: #E4F2EE; color: #0B5F52; border: 1px solid #BFE0D8; }
.chip-gap   { background: #FBE7DC; color: #9A3412; border: 1px solid #F0CDB8; }

.summary-text { margin-top: 14px; color: #374B53; font-size: 14.5px; }

.rationale {
    border-left: 3px solid #D8D4C8;
    padding: 2px 0 2px 14px;
    color: #55676E; font-style: italic; font-size: 14px; margin-top: 10px;
}

.rank-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; margin-bottom: 8px;
    background: #FFFFFF; border: 1px solid #E2E0D8; border-radius: 10px;
}
.rank-num {
    font-family: 'JetBrains Mono', monospace; font-weight: 700;
    font-size: 13px; color: #8A9AA1; margin-right: 14px;
}
.rank-score {
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 18px;
}

.point-list { margin: 0; padding-left: 18px; }
.point-list li {
    color: #374B53 !important;
    font-size: 14px; line-height: 1.75; margin-bottom: 2px;
}
.rec-text { color: #374B53 !important; font-size: 14.5px; line-height: 1.6; margin: 0; }

/* --- Force readable text regardless of the viewer's light/dark theme --- */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong {
    color: #12303A !important;
}

/* Bordered containers (the candidate cards) stay white */
.stApp [data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 14px;
}

/* Tab labels */
.stApp [data-testid="stTabs"] button p { color: #55676E; }
.stApp [data-testid="stTabs"] button[aria-selected="true"] p { color: #0E7C6B; }

/* Sidebar */
.stApp [data-testid="stSidebar"] { background: #FFFFFF; }
.stApp [data-testid="stSidebar"] * { color: #12303A !important; }

/* Keep our own coloured elements from being overridden by the rule above */
.stApp .chip-match { color: #0B5F52; }
.stApp .chip-gap   { color: #9A3412; }
.stApp .stamp-hire  { color: #0E7C6B; }
.stApp .stamp-maybe { color: #B07908; }
.stApp .stamp-pass  { color: #C2410C; }
.stApp .sec-label  { color: #8A9AA1; }
.stApp .cand-file  { color: #8A9AA1; }
.stApp .rank-num   { color: #8A9AA1; }
.stApp .rationale  { color: #55676E; }
.stApp .hero p     { color: #55676E; }
.stApp .summary-text { color: #374B53; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def score_color(score):
    if score >= 75:
        return "#0E7C6B"
    if score >= 50:
        return "#B07908"
    return "#C2410C"


def gauge_svg(score, size=118):
    """Circular score gauge."""
    color = score_color(score)
    r = 48
    circ = 2 * 3.14159 * r
    filled = circ * max(0, min(100, score)) / 100
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r="{r}" fill="none" stroke="#EAE7DE" stroke-width="10"/>
      <circle cx="60" cy="60" r="{r}" fill="none" stroke="{color}" stroke-width="10"
              stroke-linecap="round" stroke-dasharray="{filled} {circ}"
              transform="rotate(-90 60 60)"/>
      <text x="60" y="58" text-anchor="middle" font-family="JetBrains Mono, monospace"
            font-size="30" font-weight="700" fill="#12303A">{score}</text>
      <text x="60" y="76" text-anchor="middle" font-family="Inter, sans-serif"
            font-size="11" fill="#8A9AA1">/ 100</text>
    </svg>
    """


def stamp_html(verdict):
    v = (verdict or "").strip().lower()
    if v.startswith("hire"):
        cls = "stamp-hire"
    elif v.startswith("maybe") or v.startswith("consider"):
        cls = "stamp-maybe"
    else:
        cls = "stamp-pass"
    return f'<span class="stamp {cls}">{verdict}</span>'


def bullets(items):
    """Render a list as HTML so the text colour never depends on the theme."""
    if not items:
        return '<p class="rec-text">None identified</p>'
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<ul class="point-list">{lis}</ul>'


def chips(items, kind):
    if not items:
        return '<span style="color:#8A9AA1;font-size:13px;">None identified</span>'
    cls = "chip-match" if kind == "match" else "chip-gap"
    return "".join(f'<span class="chip {cls}">{i}</span>' for i in items)


# ===============================================================
#  Piece 1 — the evaluation schema
# ===============================================================
class ResumeEvaluation(BaseModel):
    candidate_name: str = Field(description="The candidate's full name as written in the resume")
    match_score: int = Field(description="Overall match score from 0 to 100")
    verdict: str = Field(description="One of exactly: Hire, Maybe, or Pass")
    candidate_summary: str = Field(description="2-3 sentence summary of the candidate")
    rationale: str = Field(description="1-2 sentences explaining why this score was given")
    matching_skills: List[str] = Field(description="Skills in the resume that match the JD")
    missing_skills: List[str] = Field(description="Important JD skills missing from the resume")
    strengths: List[str] = Field(description="Key strengths for this role")
    weaknesses: List[str] = Field(description="Key gaps or weaknesses for this role")
    recommendation: str = Field(description="Hiring recommendation with a short justification")


# ===============================================================
#  Piece 2 — the output parser
# ===============================================================
parser = PydanticOutputParser(pydantic_object=ResumeEvaluation)


# ===============================================================
#  Piece 3 — the prompts
# ===============================================================
eval_prompt = ChatPromptTemplate.from_template("""
You are an expert technical recruiter. Evaluate the candidate's resume
against the job description below.

Use ONLY the resume context provided. Do not invent any information
that is not present in the resume. If something is not in the resume,
treat it as missing.

Job Description:
{jd}

Resume Context:
{context}

{format_instructions}
""")

qa_prompt = ChatPromptTemplate.from_template("""
Answer the question using only the below context from the candidate's resume.

{context}

Question: {question}

If the answer is not in the resume, say so plainly.
""")

compare_prompt = ChatPromptTemplate.from_template("""
You are a recruiter comparing two candidates for the same job.

Job Description:
{jd}

Candidate A - {name_a} (score {score_a}):
{summary_a}
Matching skills: {match_a}
Missing skills: {miss_a}

Candidate B - {name_b} (score {score_b}):
{summary_b}
Matching skills: {match_b}
Missing skills: {miss_b}

In 3-4 sentences, say which candidate is the stronger fit and why.
Be concrete about the trade-offs.
""")

rank_prompt = ChatPromptTemplate.from_template("""
You are a recruiter making a final shortlist decision.

Job Description:
{jd}

Candidates (ranked by match score):
{candidates}

In 3-4 sentences, recommend the best candidate and justify the choice.
Mention the runner-up and why they came second.
""")


# ===============================================================
#  Sidebar — Configuration
# ===============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    provider = st.selectbox("LLM Provider", ["OpenAI"],
                            help="The model provider used for evaluation.")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Paste your key (sk-...). If left blank, the key in your .env file is used.",
    )

    model_name = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

    st.caption(
        f"**Embeddings:** `{EMBEDDING_MODEL}` (OpenAI)  \n"
        "**Vector store:** FAISS (in-memory, one per resume)  \n"
        "**Retriever:** top-k = 5"
    )

    st.divider()

    if st.button("🗑️ Clear all data / start over", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# resolve the key: sidebar first, then .env
api_key = api_key_input.strip() or os.getenv("OPENAI_API_KEY", "")


def get_llm():
    return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)


# ===============================================================
#  Header
# ===============================================================
st.markdown(
    """
    <div class="hero">
      <h1>🧭 AI Resume Screening Assistant</h1>
      <p>LangChain + RAG (FAISS) resume screener — upload resumes, paste a job description,
         and get grounded, structured candidate evaluations in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===============================================================
#  Inputs
# ===============================================================
left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="sec-label">Job Description</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Job description",
        height=260,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

with right:
    st.markdown('<div class="sec-label">Resumes</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload resumes",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} resume(s) ready to screen.")


# ===============================================================
#  Evaluate
# ===============================================================
def evaluate_one(pdf_path, jd, llm):
    """RAG evaluation of a single resume."""
    vectorstore = build_vectorstore(pdf_path, api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke(jd)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = eval_prompt | llm | parser
    result = chain.invoke({
        "jd": jd,
        "context": context,
        "format_instructions": parser.get_format_instructions(),
    })
    return result, vectorstore


if st.button("🚀 Evaluate resumes against JD", type="primary", use_container_width=True):

    if not api_key:
        st.error("Add your OpenAI API key in the sidebar to continue.")
    elif not jd_text.strip():
        st.error("Paste a job description to continue.")
    elif not uploaded_files:
        st.error("Upload at least one resume (PDF) to continue.")
    else:
        llm = get_llm()
        progress = st.progress(0.0, text="Starting...")

        evaluations = {}
        stores = {}

        for i, uploaded in enumerate(uploaded_files):
            label = uploaded.name.replace(".pdf", "")
            progress.progress(i / len(uploaded_files), text=f"Screening {label}...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                evaluation, store = evaluate_one(tmp_path, jd_text, llm)
                evaluations[label] = evaluation
                stores[label] = store
            except Exception as e:
                st.error(f"Could not screen {label}: {e}")

        progress.progress(1.0, text="Done.")

        st.session_state.evaluations = evaluations
        st.session_state.stores = stores
        st.session_state.jd_text = jd_text


# ===============================================================
#  Results
# ===============================================================
if "evaluations" in st.session_state and st.session_state.evaluations:

    evaluations = st.session_state.evaluations
    stores = st.session_state.stores
    jd_saved = st.session_state.jd_text

    st.success(f"Evaluated {len(evaluations)} resume(s).")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Individual Evaluations", "⚖️ Compare Two", "🏆 Rank & Recommend", "💬 Ask a Resume"]
    )

    # ---------- Tab 1: individual ----------
    with tab1:
        for label, ev in evaluations.items():

            with st.container(border=True):

                head_l, head_r = st.columns([3, 1])

                with head_l:
                    st.markdown(
                        f'<p class="cand-name">{ev.candidate_name}</p>'
                        f'<p class="cand-file">{label}.pdf</p>'
                        f'<div style="margin-top:10px;">{stamp_html(ev.verdict)}</div>'
                        f'<p class="summary-text">{ev.candidate_summary}</p>'
                        f'<div class="rationale">{ev.rationale}</div>',
                        unsafe_allow_html=True,
                    )

                with head_r:
                    st.markdown(gauge_svg(ev.match_score), unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="sec-label">✅ Matching skills</div>', unsafe_allow_html=True)
                    st.markdown(chips(ev.matching_skills, "match"), unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="sec-label">⚠️ Missing skills</div>', unsafe_allow_html=True)
                    st.markdown(chips(ev.missing_skills, "gap"), unsafe_allow_html=True)

                c3, c4 = st.columns(2)
                with c3:
                    st.markdown('<div class="sec-label">Strengths</div>', unsafe_allow_html=True)
                    st.markdown(bullets(ev.strengths), unsafe_allow_html=True)
                with c4:
                    st.markdown('<div class="sec-label">Weaknesses</div>', unsafe_allow_html=True)
                    st.markdown(bullets(ev.weaknesses), unsafe_allow_html=True)

                st.markdown('<div class="sec-label">Recommendation</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="rec-text">{ev.recommendation}</p>', unsafe_allow_html=True)

    # ---------- Tab 2: compare two ----------
    with tab2:
        names = list(evaluations.keys())

        if len(names) < 2:
            st.info("Upload at least two resumes to compare.")
        else:
            cc1, cc2 = st.columns(2)
            pick_a = cc1.selectbox("Candidate A", names, index=0)
            pick_b = cc2.selectbox("Candidate B", names, index=1)

            if pick_a == pick_b:
                st.warning("Pick two different candidates.")
            else:
                ev_a = evaluations[pick_a]
                ev_b = evaluations[pick_b]

                gc1, gc2 = st.columns(2)
                for col, ev in ((gc1, ev_a), (gc2, ev_b)):
                    with col:
                        with st.container(border=True):
                            st.markdown(
                                f'<div style="text-align:center;">'
                                f'<p class="cand-name">{ev.candidate_name}</p>'
                                f'{gauge_svg(ev.match_score)}'
                                f'<div>{stamp_html(ev.verdict)}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown('<div class="sec-label">✅ Matching</div>', unsafe_allow_html=True)
                            st.markdown(chips(ev.matching_skills, "match"), unsafe_allow_html=True)
                            st.markdown('<div class="sec-label">⚠️ Missing</div>', unsafe_allow_html=True)
                            st.markdown(chips(ev.missing_skills, "gap"), unsafe_allow_html=True)

                if st.button("Compare these two", use_container_width=True):
                    chain = compare_prompt | get_llm() | StrOutputParser()
                    verdict = chain.invoke({
                        "jd": jd_saved,
                        "name_a": ev_a.candidate_name, "score_a": ev_a.match_score,
                        "summary_a": ev_a.candidate_summary,
                        "match_a": ", ".join(ev_a.matching_skills),
                        "miss_a": ", ".join(ev_a.missing_skills),
                        "name_b": ev_b.candidate_name, "score_b": ev_b.match_score,
                        "summary_b": ev_b.candidate_summary,
                        "match_b": ", ".join(ev_b.matching_skills),
                        "miss_b": ", ".join(ev_b.missing_skills),
                    })
                    st.markdown('<div class="sec-label">Verdict</div>', unsafe_allow_html=True)
                    st.info(verdict)

    # ---------- Tab 3: rank & recommend ----------
    with tab3:
        ranked = sorted(evaluations.items(), key=lambda kv: kv[1].match_score, reverse=True)

        for i, (label, ev) in enumerate(ranked, start=1):
            st.markdown(
                f'<div class="rank-row">'
                f'<div style="display:flex;align-items:center;">'
                f'<span class="rank-num">#{i}</span>'
                f'<div><b>{ev.candidate_name}</b>'
                f'<div class="cand-file">{label}.pdf</div></div>'
                f'</div>'
                f'<div style="display:flex;align-items:center;gap:14px;">'
                f'{stamp_html(ev.verdict)}'
                f'<span class="rank-score" style="color:{score_color(ev.match_score)};">'
                f'{ev.match_score}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        best_label, best_ev = ranked[0]
        st.success("Top candidate: **" + best_ev.candidate_name + "** ("
                   + str(best_ev.match_score) + "/100)")

        if st.button("Generate hiring recommendation", use_container_width=True):
            lines = "\n".join(
                "- " + ev.candidate_name + ": " + str(ev.match_score) + "/100 - " + ev.verdict
                + ". Missing: " + (", ".join(ev.missing_skills) or "none")
                for _, ev in ranked
            )
            chain = rank_prompt | get_llm() | StrOutputParser()
            rec = chain.invoke({"jd": jd_saved, "candidates": lines})
            st.markdown('<div class="sec-label">Final recommendation</div>', unsafe_allow_html=True)
            st.info(rec)

    # ---------- Tab 4: ask a resume ----------
    with tab4:
        pick = st.selectbox("Choose a resume to query", list(evaluations.keys()))
        question = st.text_input(
            "Ask a question about this resume",
            placeholder="e.g. How many years of Python experience does this candidate have?",
        )

        if st.button("Ask", use_container_width=True):
            if not question.strip():
                st.warning("Type a question first.")
            else:
                retriever = stores[pick].as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in docs])

                chain = qa_prompt | get_llm() | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})

                st.markdown('<div class="sec-label">Answer</div>', unsafe_allow_html=True)
                st.info(answer)

                with st.expander("Retrieved resume context (what the model saw)"):
                    st.text(context)

else:
    st.caption("Paste a job description, upload one or more resumes, then run the screening.")
