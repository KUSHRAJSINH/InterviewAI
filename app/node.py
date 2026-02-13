"""from app.state import InterviewState
import os, time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
load_dotenv()

# --------------------------------------------------
# LLM FACTORY (OPENROUTER – FREE)
# --------------------------------------------------
def get_llm():
    return ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )

model=ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, timeout=20)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def format_qa(questions, answers):
    return "\n".join(
        f"Q{i+1}: {q}\nA{i+1}: {a}"
        for i, (q, a) in enumerate(zip(questions, answers))
    )

# --------------------------------------------------
# PROMPTS
# --------------------------------------------------
def final_evaluation_prompt(state):
    return f"""
#You are a senior technical interviewer.

#Candidate name:
#{state['candidate_name']}

#Interview questions and answers:
#{format_qa(state['question_history'], state['answer_history'])}

#Evaluate the candidate holistically.

#Provide:
#- Overall score (out of 10)
#- Strengths
#- Weaknesses
#- Topic gaps
#- Communication quality
#- Technical depth
#- Hiring recommendation
#- Short summary
"""

def topic_extraction_prompt(answer):
    return f"""
#Extract the main technical topic from the answer.
#Return ONLY 1–2 words.

#Answer:
#{answer}
"""

# --------------------------------------------------
# NODES
# --------------------------------------------------
def load_resume(state: InterviewState):
    loader = PyPDFLoader(state["resume_path"])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    state["resume_vectorstore"] = vectorstore
    return state

def extract_candidate_name(state):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        timeout=30
    )

#    prompt = """
#Extract the candidate's full name from this resume text.
#If not found, return "Candidate".
#ONLY return the name.
"""

    docs = state["resume_vectorstore"].similarity_search(
        "candidate name", k=3
    )

    text = "\n".join(d.page_content for d in docs)

    name = llm.invoke(prompt + "\n\n" + text).content.strip()

    state["candidate_name"] = name or "Candidate"
    return state

def generate_question(state: InterviewState):
    #llm = get_llm()
    #state.setdefault("question_count", 0)
    #if "question_count" not in state:
     #   state["question_count"] = 0
     
     
     
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        timeout=30
    )

    prompt = f"""
#You are an AI interviewer.

#Based on the candidate's resume, ask ONE strong technical interview question.
#Start medium difficulty.
#Ask only the question, nothing else.
"""
 
    docs = state["resume_vectorstore"].similarity_search(
        "skills experience projects", k=4
    )

    resume_context = "\n".join(d.page_content for d in docs)

    question = llm.invoke(prompt + "\n\n" + resume_context).content.strip()
 
    state["current_question"] = question
    state["question_history"].append(question)
    state["phase"] = "question"
    state["depth_level"] = 1
    #state["question_count"] += 1

    return state



def evaluate_answer(state: InterviewState):
    topic = extract_topic_from_answer(state["current_answer"])

    if topic == state["current_topic"]:
        state["depth_level"] += 1
    else:
        state["current_topic"] = topic
        state["depth_level"] = 1

    if state["question_count"] >= state["max_questions"]:
        state["phase"] = "FINAL"
    else:
        state["phase"] = "question"

    return state


def extract_topic_from_answer(answer):
    #llm = get_llm()
    
    model=ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, timeout=20)

    result = model.invoke(topic_extraction_prompt(answer))
    return result.content.strip().lower()

def get_user_answer(state: InterviewState):
    answer = state["current_answer"]
    state["answer_history"].append(answer)

    topic = extract_topic_from_answer(answer)

    if topic == state["current_topic"]:
        state["depth_level"] += 1
    else:
        state["current_topic"] = topic
        state["depth_level"] = 0

    return state

def time_and_phase_check(state: InterviewState):
    elapsed = time.time() - state["interview_start_time"]

    if elapsed >= state["max_duration_seconds"]:
        state["phase"] = "FINAL"
    elif state["depth_level"] >= 3:
        state["phase"] = "ASK"
        state["depth_level"] = 0
        state["current_topic"] = ""
    else:
        state["phase"] = "FOLLOW_UP"

    return state

def final_evaluation(state: InterviewState):
    #llm = get_llm()
    model=ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, timeout=20)
 
    result = model.invoke(final_evaluation_prompt(state))
    state["final_evaluation"] = result.content.strip()
    return state
"""


import time
import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from app.state import InterviewState

load_dotenv()


# ==========================================================
# Resume Loader
# ==========================================================
def load_resume(state: InterviewState):
    loader = PyPDFLoader(state["resume_path"])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80
    )
    chunks = splitter.split_documents(docs)

    state["resume_vectorstore"] = FAISS.from_documents(
        chunks,
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    return state


# ==========================================================
# Candidate Name Extractor
# ==========================================================
def extract_candidate_name(state: InterviewState):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    docs = state["resume_vectorstore"].similarity_search(
        "candidate name", k=3
    )
    text = "\n".join(d.page_content for d in docs)

    prompt = """
Extract the candidate's full name.
If not found, return "Candidate".
ONLY return the name.
"""

    name = llm.invoke(prompt + "\n\n" + text).content.strip()
    state["candidate_name"] = name or "Candidate"
    return state


# ==========================================================
# Topic Extractor
# ==========================================================
def extract_topic_from_answer(answer: str) -> str:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = f"""
Extract the main technical topic from this answer.
Return ONLY 1–2 words.

Answer:
{answer}
"""
    return llm.invoke(prompt).content.strip().lower()


# ==========================================================
# Evaluate Answer Quality (Structured)
# ==========================================================
def evaluate_answer_quality(state: InterviewState):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    latest_answer = state["answer_history"][-1]

    prompt = f"""
You are a senior technical interviewer.

Evaluate the candidate's answer.

Question:
{state['current_question']}

Answer:
{latest_answer}

Return ONLY valid JSON in this format:

{{
  "quality_score": 0-10,
  "depth_score": 0-10,
  "clarity_score": 0-10,
  "confidence_level": "low | medium | high",
  "weak_areas": ["..."],
  "follow_up_required": true/false
}}

Rules:
- If answer is vague → low depth
- If missing examples → weak clarity
- If incomplete → follow_up_required = true
- No explanation outside JSON
"""

    result = llm.invoke(prompt).content.strip()

    try:
        data = json.loads(result)
    except:
        # fallback if JSON fails
        data = {
            "quality_score": 5,
            "depth_score": 5,
            "clarity_score": 5,
            "confidence_level": "medium",
            "weak_areas": [],
            "follow_up_required": False
        }

    state["quality_score"] = data["quality_score"]
    state["depth_score"] = data["depth_score"]
    state["clarity_score"] = data["clarity_score"]
    state["confidence_level"] = data["confidence_level"]
    state["weak_areas"] = data["weak_areas"]
    state["follow_up_required"] = data["follow_up_required"]

    # -----------------------------
    # Topic Extraction
    # -----------------------------
    topic = extract_topic_from_answer(latest_answer)
    state["current_topic"] = topic

    # -----------------------------
    # Topic Mastery Tracking
    # -----------------------------
    if "topic_mastery" not in state:
        state["topic_mastery"] = {}

    if topic:
        if topic not in state["topic_mastery"]:
            state["topic_mastery"][topic] = []

        state["topic_mastery"][topic].append(state["quality_score"])

    return state


# ==========================================================
# Dynamic Difficulty Engine
# ==========================================================
def adjust_difficulty(state: InterviewState):

    if "difficulty_level" not in state:
        state["difficulty_level"] = 2  # Start medium

    if state["quality_score"] >= 8:
        state["difficulty_level"] = min(4, state["difficulty_level"] + 1)

    elif state["quality_score"] <= 4:
        state["difficulty_level"] = max(1, state["difficulty_level"] - 1)

    return state


# ==========================================================
# Adaptive Question Generator
# ==========================================================
def generate_question(state: InterviewState):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4
    )

    docs = state["resume_vectorstore"].similarity_search(
        "skills experience projects", k=4
    )
    context = "\n".join(d.page_content for d in docs)

    previous_questions = state.get("question_history", [])

    prompt = f"""
You are an adaptive AI interviewer.

Current Topic: {state.get('current_topic', '')}
Difficulty Level: {state.get('difficulty_level', 2)}
Weak Areas: {state.get('weak_areas', [])}
Follow Up Required: {state.get('follow_up_required', False)}

Resume Context:
{context}

Instructions:

- If Follow Up Required is True:
  Ask a deeper follow-up question on SAME topic.
  Focus specifically on weak areas.

- Else:
  Move to a NEW technical skill from resume.
  Avoid repeating previous topics.

Difficulty meaning:
1 = Easy conceptual
2 = Medium practical
3 = Hard implementation
4 = Advanced system design

Rules:
- Ask ONLY ONE question.
- Do NOT include explanation.
- Do NOT repeat previous questions.
"""

    for _ in range(3):
        response = llm.invoke(prompt)
        question = response.content.strip()

        if question not in previous_questions:
            state["current_question"] = question
            return state

    state["current_question"] = question
    return state


# ==========================================================
# Final Evaluation
# ==========================================================
def format_qa(questions, answers):
    return "\n".join(
        f"Q{i+1}: {q}\nA{i+1}: {a}"
        for i, (q, a) in enumerate(zip(questions, answers))
    )


def final_evaluation(state: InterviewState):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = f"""
You are a senior technical interviewer.

Candidate Name:
{state['candidate_name']}

Interview Questions and Answers:
{format_qa(state['question_history'], state['answer_history'])}

Proctoring Report:
- Cheat Flags: {state['cheat_flags']}
- Integrity Risk Score: {state['cheat_score']}

Rules:
- If Integrity Risk Score >= 6 → HIGH RISK
- If 3–5 → MODERATE RISK
- If <= 2 → LOW RISK

Provide:
- Overall score (out of 10)
- Technical strengths
- Technical weaknesses
- Communication quality
- Integrity assessment
- Hiring recommendation
- Short summary
"""

    state["final_report"] = llm.invoke(prompt).content.strip()
    return state
