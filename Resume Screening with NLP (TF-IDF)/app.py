import streamlit as st
import pickle
import PyPDF2
import re
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="centered")

st.title("📄 AI Resume Screening System")
st.write("Target Job Role select karein aur candidate resume evaluate karein.")

# Dataset roles ke specific requirements
JOB_PROFILES = {
    "Machine Learning / AI Engineer": """
    Python scikit-learn TensorFlow Keras PyTorch deep learning computer vision NLP 
    machine learning neural networks model evaluation pandas numpy opencv
    """,
    "Data Analyst": """
    SQL data analysis Excel PowerBI Tableau data visualization pandas numpy 
    ETL data cleaning exploratory data analysis statistical modeling dashboarding
    """,
    "Software Engineer / Backend Developer": """
    Java C++ OOP data structures algorithms DSA system design REST APIs 
    SQL relational databases git linux multithreading backend microservices
    """,
    "DevOps / Cloud Engineer": """
    Docker Kubernetes CI/CD Jenkins AWS Terraform Linux bash scripting 
    Ansible cloud infrastructure Prometheus monitoring logging automation
    """
}

# 1. Text Cleaning Function
def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# 2. Load TF-IDF Vectorizer
@st.cache_resource
def load_vectorizer():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

vectorizer = load_vectorizer()

# --- Section 1: Dropdown Selection ---
st.subheader("1. Select Target Job Role")
selected_role = st.selectbox(
    "Target Job Role choose karein:",
    options=list(JOB_PROFILES.keys())
)
target_job_req = JOB_PROFILES[selected_role]

# --- Section 2: Resume Input ---
st.subheader("2. Candidate Resume")
input_choice = st.radio("Resume kaise provide karna chahte hain?", ("Upload Resume PDF", "Paste Resume Text"), horizontal=True)

resume_text = ""
if input_choice == "Upload Resume PDF":
    uploaded_file = st.file_uploader("Candidate ka PDF resume upload karein", type=["pdf"])
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            resume_text += page.extract_text() or ""
        st.success("PDF resume load ho gaya!")
else:
    resume_text = st.text_area(
        "Candidate ka Resume text yahan paste karein:",
        height=180, 
        placeholder="Skills, Projects, Education, Work Experience..."
    )

# --- Section 3: Evaluation ---
if st.button("Evaluate Match Score", type="primary"):
    if not resume_text.strip():
        st.warning("Pehle candidate ka resume paste karein ya PDF upload karein!")
    else:
        # Dono ko ALAG ALAG vector banayein (Concanetate nahi karna)
        job_clean = clean_text(target_job_req)
        resume_clean = clean_text(resume_text)
        
        job_vec = vectorizer.transform([job_clean])
        resume_vec = vectorizer.transform([resume_clean])
        
        # Calculate actual mathematical cosine match
        sim_score = cosine_similarity(job_vec, resume_vec)[0][0]
        match_percentage = min(100, int(round(sim_score * 180)))  # Scaled to 0-100%
        
        # Map to 0-4 Scale dynamically
        if match_percentage >= 75:
            score, verdict = 4, "Excellent Fit (Ideal Match)"
        elif match_percentage >= 55:
            score, verdict = 3, "Strong Fit (Highly Recommended)"
        elif match_percentage >= 35:
            score, verdict = 2, "Moderate Fit (Partial Match)"
        elif match_percentage >= 15:
            score, verdict = 1, "Below Average Fit"
        else:
            score, verdict = 0, "Low Fit (Skills Mismatch)"
            
        st.markdown("---")
        st.success(f"🎯 **Target Role:** {selected_role} — **{verdict}**")
        st.progress(match_percentage / 100)
        st.info(f"Match Level Score: {score} / 4 ({match_percentage}% Relevance Overlap)")