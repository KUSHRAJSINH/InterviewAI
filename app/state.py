"""from typing import TypedDict, List, Any


class InterviewState(TypedDict):
    # -------------------------
    # Resume & Retrieval
    # -------------------------
    resume_path: str
    candidate_name: str
    resume_vectorstore: Any

    # -------------------------
    # Current Turn
    # -------------------------
    current_question: str
    current_answer: str

    # -------------------------
    # History (IMPORTANT)
    # -------------------------
    question_history: List[str]
    answer_history: List[str]

    # -------------------------
    # Interview Control
    # -------------------------
    current_topic: str
    depth_level: int

    interview_start_time: float
    max_duration_seconds: int

    # -------------------------
    # Phase Control
    # -------------------------
    phase: str  # e.g. "question", "answer", "evaluation", "end"

    # -------------------------
    # Final Output
    # -------------------------
    final_report: str
"""


from typing import TypedDict, List, Any, Optional


class InterviewState(TypedDict):
    # Resume
    resume_path: str
    candidate_name: str
    resume_vectorstore: Any

    # Current turn
    current_question: str
    current_answer: str

    # History
    question_history: List[str]
    answer_history: List[str]

    # Topic tracking
    current_topic: str
    depth_level: int

    # Timing
    interview_start_time: float
    max_duration_seconds: int

    # Control (STREAMLIT ONLY)
    question_count: int
    max_questions: int

    # Phase
    phase: str  # question | WAIT_FINAL | FINAL

    # Final output
    final_report: str

    cheat_flags: List[str]
    cheat_score: int
    face_missing_seconds: int

    tab_switch_count: int
    tab_flagged_questions: List[int]


    quality_score: int
    depth_score: int
    clarity_score: int
    confidence_level: str

    difficulty_level: int #1-4
    weak_areas:list
    topic_mastery:dict
    follow_up_required: bool
    
    

   

 