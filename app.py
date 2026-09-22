import os
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI HR Policy Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI HR Policy Assistant")
st.write("Ask questions about company HR policies.")


# -----------------------------
# Gemini API
# -----------------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("Gemini API key is not configured.")
    st.stop()


# -----------------------------
# Load HR Policy PDFs
# -----------------------------
DOCUMENTS_PATH = "hr_documents"

chunks = []

for filename in os.listdir(DOCUMENTS_PATH):
    if filename.endswith(".pdf"):
        filepath = os.path.join(DOCUMENTS_PATH, filename)

        reader = PdfReader(filepath)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                chunks.append({
                    "text": text,
                    "source": filename,
                    "page": page_number
                })


# -----------------------------
# Create TF-IDF Vector Store
# -----------------------------
chunk_texts = [chunk["text"] for chunk in chunks]

vectorizer = TfidfVectorizer(stop_words="english")
chunk_vectors = vectorizer.fit_transform(chunk_texts)


# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------
def retrieve_chunks(query, top_k=3):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index]["text"],
            "source": chunks[index]["source"],
            "page": chunks[index]["page"],
            "score": float(scores[index])
        })

    return results


# -----------------------------
# Generate HR Answer
# -----------------------------
def generate_hr_answer(question, top_k=3, threshold=0.20):

    results = retrieve_chunks(question, top_k)

    if not results or results[0]["score"] < threshold:

        return {
            "answer": (
                "Sorry, I can only answer questions "
                "related to the provided HR policies."
            ),
            "sources": []
        }

    context = "\n\n".join([
        f"Source: {r['source']}\n"
        f"Page: {r['page']}\n"
        f"Content: {r['text']}"
        for r in results
    ])

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question ONLY using the HR policy
information provided in the context below.

Do not use outside knowledge.
Do not invent information.

If the answer is not available in the context,
say that the information is not available.

HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Give a short, clear and accurate answer.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources": results
    }


# -----------------------------
# Chat Interface
# -----------------------------
question = st.text_input(
    "Ask your HR policy question:",
    placeholder="Example: How many annual leave days do employees receive?"
)


if st.button("Ask", type="primary"):

    if question.strip():

        with st.spinner("Searching HR policies and generating answer..."):

            result = generate_hr_answer(question)

        st.subheader("Answer")
        st.write(result["answer"])

        if result["sources"]:

            st.subheader("📚 Sources")

            for source in result["sources"]:

                st.write(
                    f"**{source['source']}** | "
                    f"Page {source['page']} | "
                    f"Similarity {source['score']:.3f}"
                )

    else:
        st.warning("Please enter an HR policy question.")
