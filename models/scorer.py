from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_semantic_score(resume_text, jd_text):
    embeddings = model.encode([resume_text, jd_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(score) * 100, 2)

def get_tfidf_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return round(float(score) * 100, 2)

def get_skill_gaps(resume_text, jd_text):
    jd_words = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    tech_keywords = {'python','sql','docker','kubernetes','aws','react','flask',
                     'fastapi','mlops','tensorflow','pytorch','langchain','rag'}
    jd_skills = jd_words & tech_keywords
    missing = jd_skills - resume_words
    matched = jd_skills & resume_words
    return list(matched), list(missing)

def get_final_score(resume_text, jd_text):
    semantic = get_semantic_score(resume_text, jd_text)
    tfidf = get_tfidf_score(resume_text, jd_text)
    final = round(0.7 * semantic + 0.3 * tfidf, 1)
    matched, missing = get_skill_gaps(resume_text, jd_text)
    return {
        "score": min(final, 100),
        "semantic_score": semantic,
        "tfidf_score": tfidf,
        "matched_skills": matched,
        "missing_skills": missing
    }