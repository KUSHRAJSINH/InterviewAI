"""from app.node import *
from app.state import InterviewState
from langgraph.graph import StateGraph, START, END

'''
def graph_builder():
    graph = StateGraph(InterviewState)

    graph.add_node("resume_loader", load_resume)
    graph.add_node("content_extractor", extract_content)
    graph.add_node("question_generator", generate_question)
    graph.add_node("answer_inputter", input_answer)
    graph.add_node("answer_evaluator", evaluate_answer)

    graph.add_edge(START, "resume_loader")
    graph.add_edge("resume_loader", "content_extractor")
    graph.add_edge("content_extractor", "question_generator")
    graph.add_edge("question_generator", "answer_inputter")
    graph.add_edge("answer_inputter", "answer_evaluator")
    graph.add_edge("answer_evaluator", END)

    # 🔑 THIS IS CRITICAL
    compiled_graph = graph.compile()
    return compiled_graph
'''

'''
def graph_builder():
    graph=StateGraph(InterviewState)



    #------ nodes -------

    graph.add_node("resume_loader",load_resume)
    graph.add_node("content_extractor",extract_candidate_name)
    graph.add_node("decide_question",generate_question)
    #graph.add_node("get_answer",get_user_answer)
    graph.add_node("time_phase_check",time_and_phase_check)
    graph.add_node("final_evaluation",final_evaluation)




    #----------------entry------


    graph.add_edge(START,"resume_loader")
    graph.add_edge("resume_loader","content_extractor")
    graph.add_edge("content_extractor","decide_question")

  #------interview loop ------

    #graph.add_edge("decide_question","get_answer")
    #graph.add_edge("get_answer","time_phase_check")


#--------conditional LOOP------------------------
    graph.add_conditional_edges(
        "time_phase_check",
        lambda state: "continue" if state['phase'] != "FINAL" else "end",
        {
            "continue": "decide_question",
            "end": "final_evaluation"
        }
    )


    graph.add_edge("final_evaluation",END)

    return graph.compile()

'''
"""
"""
from langgraph.graph import StateGraph, START, END
from app.state import InterviewState

from app.node import *

def build_init_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("resume_loader", load_resume)
    graph.add_node("content_extractor", extract_candidate_name)
    graph.add_node("first_question", generate_question)

    graph.add_edge(START, "resume_loader")
    graph.add_edge("resume_loader", "content_extractor")
    graph.add_edge("content_extractor", "first_question")
    graph.add_edge("first_question", END)

    return graph.compile()
""""""
from langgraph.graph import StateGraph, START, END
from app.state import InterviewState
from app.node import (
    load_resume,
    extract_candidate_name,
    generate_question,
    final_evaluation,
    extract_topic_from_answer,
)
import time

def evaluate_answer_node(state: InterviewState):
    answer = state["current_answer"]
    topic = extract_topic_from_answer(answer)

    if topic == state["current_topic"]:
        state["depth_level"] += 1
    else:
        state["current_topic"] = topic
        state["depth_level"] = 1

    # stop after 5 questions or time over
    elapsed = time.time() - state["interview_start_time"]

    if (
        elapsed >= state["max_duration_seconds"]
        or state["question_count"] >= state["max_questions"]
    ):
        state["phase"] = "FINAL"
    else:
        state["phase"] = "question"

    return state


def graph_builder():
    graph = StateGraph(InterviewState)

    graph.add_node("resume_loader", load_resume)
    graph.add_node("content_extractor", extract_candidate_name)
    graph.add_node("question_generator", generate_question)
    graph.add_node("answer_evaluator", evaluate_answer_node)
    graph.add_node("final_evaluation", final_evaluation)

    graph.add_edge(START, "resume_loader")
    graph.add_edge("resume_loader", "content_extractor")
    graph.add_edge("content_extractor", "question_generator")
    graph.add_edge("question_generator", END)

    graph.add_edge("answer_evaluator", END)
    graph.add_edge("final_evaluation", END)

    return graph.compile()
"""

"""
from langgraph.graph import StateGraph, START, END
from app.state import InterviewState
from app.node import (
    load_resume,
    extract_candidate_name,
    generate_question,
    final_evaluation,
)

# -------------------------
# Question Graph
# -------------------------
def question_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("resume_loader", load_resume)
    graph.add_node("name_extractor", extract_candidate_name)
    graph.add_node("question_generator", generate_question)

    graph.add_edge(START, "resume_loader")
    graph.add_edge("resume_loader", "name_extractor")
    graph.add_edge("name_extractor", "question_generator")
    graph.add_edge("question_generator", END)

    return graph.compile()


# -------------------------
# Final Evaluation Graph
# -------------------------
def final_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("final_evaluation", final_evaluation)
    graph.add_edge(START, "final_evaluation")
    graph.add_edge("final_evaluation", END)

    return graph.compile()
"""


from langgraph.graph import StateGraph, START, END
from app.state import InterviewState
from app.node import (
    load_resume,
    extract_candidate_name,
    generate_question,
    final_evaluation,
)

# ==========================================================
# Initial Question Graph (ONLY FOR START)
# ==========================================================
def init_interview_graph():
    """
    Used only once when interview starts.
    Loads resume, extracts name, generates first question.
    """

    graph = StateGraph(InterviewState)

    graph.add_node("resume_loader", load_resume)
    graph.add_node("name_extractor", extract_candidate_name)
    graph.add_node("first_question_generator", generate_question)

    graph.add_edge(START, "resume_loader")
    graph.add_edge("resume_loader", "name_extractor")
    graph.add_edge("name_extractor", "first_question_generator")
    graph.add_edge("first_question_generator", END)

    return graph.compile()


# ==========================================================
# Final Evaluation Graph
# ==========================================================
def final_graph():
    """
    Used once after interview completion.
    """

    graph = StateGraph(InterviewState)

    graph.add_node("final_evaluation", final_evaluation)

    graph.add_edge(START, "final_evaluation")
    graph.add_edge("final_evaluation", END)

    return graph.compile()
