import streamlit as st
import pandas as pd
from analyzer import ResumeAnalyzer
from skill_extractor import SkillExtractor
import PyPDF2
import altair as alt

st.set_page_config(page_title="Resume Skill Gap Analyzer", page_icon="🎯", layout="wide")

# ── Custom Professional CSS ─────────────────────────
st.markdown("""
<style>
/* Page background */
.main { background-color: #0d1117; }

/* Section header styling */
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 6px;
}

/* Report card */
.report-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 28px;
}

/* Candidate name header */
.candidate-name {
    font-size: 24px;
    font-weight: 800;
    color: #58a6ff;
    margin-bottom: 4px;
}

/* Section sub-header */
.report-section-title {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #8b949e;
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 16px;
}

/* Metric box */
.metric-box {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}
.metric-label {
    font-size: 12px;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #e6edf3;
}

/* Skill badge */
.skill-badge {
    display: inline-block;
    padding: 5px 14px;
    margin: 4px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #30363d;
    margin: 20px 0;
}

/* Recommendation box */
.rec-box {
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 14px;
    font-weight: 600;
    margin-top: 10px;
}

/* Expander header color */
.streamlit-expanderHeader {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer():
    """Load models once to save time."""
    return ResumeAnalyzer(skill_extractor=SkillExtractor())

analyzer = load_analyzer()

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def skill_badges(skills, hex_color):
    """Renders skills as beautiful HTML badges."""
    if not skills:
        return "<p style='color: gray; font-style: italic; margin-top: 5px;'>None detected</p>"
    
    html = "<div style='margin-top: 10px; margin-bottom: 10px;'>"
    for s in skills:
        html += f"<span style='display:inline-block; padding: 4px 12px; margin: 3px; border-radius: 15px; font-size: 13px; font-weight: 600; color: white; background-color: {hex_color}; box-shadow: 1px 1px 3px rgba(0,0,0,0.2);'>{s.upper()}</span>"
    html += "</div>"
    return html

st.title("🎯 Resume Skill Gap Analyzer")
st.markdown("Automate Resume Screening using **NLP (spaCy)** and **TF-IDF + Cosine Similarity**.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Job Description")
    jd_text = st.text_area("Paste the Job Description here...", height=180, placeholder="We are looking for a Machine Learning Engineer with experience in Python, PyTorch, and NLP...")

with col2:
    st.subheader("📄 Resumes")
    uploaded_files = st.file_uploader("Upload Resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("🚀 Generate Professional Report", use_container_width=True, type="primary"):
    if not jd_text.strip():
        st.warning("⚠️ Please paste the Job Description.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        with st.spinner("🧠 Extracting entities and computing vector similarities..."):
            resumes_dict = {}
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(file)
                else:
                    text = file.read().decode("utf-8")
                resumes_dict[file.name] = text
            
            results = analyzer.analyse(jd_text, resumes_dict)
            st.success("✅ Automated AI Analysis Complete!")
            
            # --- LEADERBOARD ---
            st.markdown("### 🏆 Applicant Leaderboard")
            
            # Clean leaderboard: Remove huge comma-separated strings so the table looks neat.
            clean_data = []
            for r in results:
                clean_data.append({
                    "Candidate": r.resume_name,
                    "Match Score (%)": round(r.similarity_score * 100, 2),
                    "Overall Verdict": r.match_category,
                    "Skills Matched": len(r.matched_skills),
                    "Skill Gaps": len(r.missing_skills)
                })
            df_clean = pd.DataFrame(clean_data)
            
            st.dataframe(
                df_clean,
                use_container_width=True,
                column_config={
                    "Match Score (%)": st.column_config.ProgressColumn(
                        "Match Score (%)",
                        help="Candidate's similarity score",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    )
                }
            )
            
            st.divider()
            
            # --- DETAILED PROFESSIONAL REPORT ---
            st.markdown("### 🔍 In-Depth Candidate Reports")
            st.caption("Expand each candidate to see a professional skill gap analysis, AI extracted keywords, and hiring recommendations.")
            
            for i, res in enumerate(results):
                score_val = round(res.similarity_score * 100, 2)

                # ── Determine score color ──
                if score_val >= 75:
                    score_color = "#2ea043"
                    verdict_bg = "#0d2d1a"
                    verdict_border = "#2ea043"
                elif score_val >= 55:
                    score_color = "#3fb950"
                    verdict_bg = "#0d2d1a"
                    verdict_border = "#3fb950"
                elif score_val >= 35:
                    score_color = "#d29922"
                    verdict_bg = "#2d1f0a"
                    verdict_border = "#d29922"
                else:
                    score_color = "#f85149"
                    verdict_bg = "#2d0d0a"
                    verdict_border = "#f85149"

                # ─────────────────────────────────────
                # REPORT CARD — Always visible
                # ─────────────────────────────────────
                st.markdown(f"""
                <div class="report-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                        <div>
                            <div class="candidate-name">📄 {res.resume_name}</div>
                            <div style="color:#8b949e; font-size:13px; margin-top:2px;">Candidate #{i+1} &nbsp;·&nbsp; {res.match_category}</div>
                        </div>
                        <div style="background:{verdict_bg}; border:2px solid {score_color}; border-radius:12px; padding:10px 24px; text-align:center;">
                            <div style="font-size:11px; color:{score_color}; font-weight:700; letter-spacing:1px; text-transform:uppercase;">Match Score</div>
                            <div style="font-size:36px; font-weight:900; color:{score_color};">{score_val}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Metric Row ──
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🎯 Match Score", f"{score_val}%")
                m2.metric("✅ Skills Matched", len(res.matched_skills))
                m3.metric("❌ Critical Gaps", len(res.missing_skills))
                m4.metric("🌟 Bonus Skills", len(res.extra_skills))

                st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

                # ── Skills Section ──
                col_m, col_g, col_b = st.columns(3)

                with col_m:
                    st.markdown("<div class='report-section-title'>✅ Matched Skills</div>", unsafe_allow_html=True)
                    st.markdown(skill_badges(res.matched_skills, "#196c2e"), unsafe_allow_html=True)

                with col_g:
                    st.markdown("<div class='report-section-title'>❌ Missing Skills (Gaps)</div>", unsafe_allow_html=True)
                    st.markdown(skill_badges(res.missing_skills, "#b91c1c"), unsafe_allow_html=True)

                with col_b:
                    st.markdown("<div class='report-section-title'>🌟 Bonus / Extra Skills</div>", unsafe_allow_html=True)
                    st.markdown(skill_badges(res.extra_skills, "#1d4ed8"), unsafe_allow_html=True)

                st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

                # ── Hiring Recommendation ──
                st.markdown("<div class='report-section-title'>💡 AI Hiring Recommendation</div>", unsafe_allow_html=True)

                if "Excellent" in res.match_category:
                    st.success("🔥 **STRONG HIRE** — Candidate exceeds the minimum skill threshold and covers most of the required technical competencies. Recommend proceeding directly to technical interview.")
                elif "Good" in res.match_category:
                    st.success("👍 **RECOMMENDED** — Solid technical profile with good alignment to the JD. Minor upskilling may be needed for the missing skills listed above. Worth interviewing.")
                elif "Fair" in res.match_category:
                    st.warning(f"⚠️ **BORDERLINE** — Candidate is missing {len(res.missing_skills)} key skills. Review their bonus skills to assess transferability. A screening call is advised before proceeding further.")
                else:
                    st.error(f"❌ **NOT RECOMMENDED** — Large skill gap of {len(res.missing_skills)} critical skills detected. Candidate profile does not sufficiently align with the technical requirements of this role.")

                st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

                # ── TF-IDF Bar Chart ──
                st.markdown("<div class='report-section-title'>📊 Top Keywords in Resume (TF-IDF Analysis)</div>", unsafe_allow_html=True)
                st.caption("Statistically significant terms from the candidate's resume, revealing their areas of technical focus and expertise.")

                if res.top_tfidf_terms:
                    df_chart = pd.DataFrame(res.top_tfidf_terms, columns=["Keyword", "Relevance Score"])
                    chart = alt.Chart(df_chart).mark_bar(cornerRadiusEnd=5).encode(
                        x=alt.X('Relevance Score:Q', title="TF-IDF Relevance Weight", axis=alt.Axis(grid=False)),
                        y=alt.Y('Keyword:N', sort='-x', title=""),
                        color=alt.Color('Relevance Score:Q', scale=alt.Scale(scheme='purples'), legend=None),
                        tooltip=['Keyword', 'Relevance Score']
                    ).properties(height=300).configure_view(strokeWidth=0).configure_axis(labelColor='#c9d1d9', titleColor='#8b949e')
                    st.altair_chart(chart, use_container_width=True)

                # Separator between candidates
                st.markdown("<br><br>", unsafe_allow_html=True)
