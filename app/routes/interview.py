from fastapi import APIRouter,UploadFile,File, Form
#from app.edges import question_graph, final_graph
import shutil
import os
import time 
import uuid
from app.edges import init_interview_graph,final_graph      
from app.node import (
    evaluate_answer_quality,
    adjust_difficulty,
    generate_question
)

MAX_CHEAT_SCORE = 8

router=APIRouter()

#simple in memory session storage 


sessions={}

@router.post("/start-interview")
async def start_interview(file: UploadFile=File(...)):
    os.makedirs("data",exist_ok=True)
    file_path=f"data/{uuid.uuid4()}.pdf"

    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    state={
        "resume_path":file_path,
        "candidate_name":"",
        "resume_vectorstore":None,

        "current_question":"",
        "current_answer":"",

        "question_history": [],
        "answer_history": [],

        "current_topic": "",
        "depth_level": 0,

        "interview_start_time": time.time(),
        "max_duration_seconds": 900,

        "question_count": 0,
        "max_questions": 5,

        "phase": "question",
        "final_report": "",

        "cheat_flags": [],
        "cheat_score": 0,
    }


    graph=init_interview_graph()
    state=graph.invoke(state)


    session_id=str(uuid.uuid4())
    sessions[session_id]=state


    return{
        "session_id":session_id,
        "question":state['current_question']
    }

@router.post("/submit-answer")
async def submit_answer(
    session_id: str = Form(...),
    answer: str = Form(...)
):
    state = sessions.get(session_id)

    if not state:
        return {"error": "Invalid session"}

    # Store answer
    state["answer_history"].append(answer)
    state["question_history"].append(state["current_question"])
    state["question_count"] += 1

    # -----------------------------
    # TIME CHECK
    # -----------------------------
    elapsed_time = time.time() - state["interview_start_time"]

    if (
        elapsed_time >= state["max_duration_seconds"]
        or state["question_count"] >= state["max_questions"]
    ):
        state["phase"] = "FINAL"
        sessions[session_id] = state
        return {"status": "completed"}

    # -----------------------------
    # ADAPTIVE FLOW
    # -----------------------------
    state = evaluate_answer_quality(state)
    state = adjust_difficulty(state)
    state = generate_question(state)

    sessions[session_id] = state

    return {"question": state["current_question"]}



# -------------------------
# REPORT CHEAT (NEW)
# -------------------------
@router.post("/report-cheat")
async def report_cheat(
    session_id: str = Form(...),
    event: str = Form(...)
):
    state = sessions.get(session_id)

    if not state:
        return {"error": "Invalid session"}

    # Store event
    state["cheat_flags"].append(event)

    # Weighted scoring logic
    event_lower = event.lower()

    if "tab" in event_lower:
        state["cheat_score"] += 2
        state["tab_switch_count"] += 1

    elif "paste" in event_lower:
        state["cheat_score"] += 3

    elif "copy" in event_lower:
        state["cheat_score"] += 2

    elif "window" in event_lower:
        state["cheat_score"] += 1

    else:
        state["cheat_score"] += 1

    sessions[session_id] = state

    return {
    "status": "recorded",
    "current_cheat_score": state["cheat_score"],
    "terminated": state["cheat_score"] >= MAX_CHEAT_SCORE
    }
 



@router.post("/final-report")
async def final_report(session_id: str = Form(...)):

    print("FINAL REPORT CALLED")

    state = sessions.get(session_id)

    if not state:
        print("Invalid session")
        return {"error": "Invalid session"}

    print("Questions:", state.get("question_history"))
    print("Answers:", state.get("answer_history"))

    graph = final_graph()
    state = graph.invoke(state)

    print("Final report generated")

    sessions[session_id] = state

    return {"report": state.get("final_report", "No report")}
