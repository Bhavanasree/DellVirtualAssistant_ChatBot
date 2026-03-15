
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import build_vector_db, generate_text_summaries, get_vector_store, get_reranker, create_rag_agent
from ragas_evaluator import evaluate_rag

def run_evaluation():
    # Build the vector database
    text_chunks, tables = build_vector_db()

    # Generate summaries
    text_summaries, table_summaries = generate_text_summaries(text_chunks, tables)

    # Get vector store and retriever
    retriever, vector_store = get_vector_store(text_chunks, tables, text_summaries, table_summaries)

    # Get reranker
    compression_retriever = get_reranker(retriever)

    # Create RAG agent
    agent = create_rag_agent(vector_store, compression_retriever)

    questions = [
        "Which Dell laptop is best for gaming?",
        "What are the specs of Dell XPS 13?"
    ]

    ground_truths = [
        "Alienware series is best for gaming",
        "Dell XPS 13 has Intel processor and high resolution display"
    ]

    answers = []
    contexts = []

    for q in questions:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": q}]}
        )

        answers.append(response["messages"][-1].content)

        # For contexts, we need to extract from the tool calls or retriever
        # This is a placeholder; in a real scenario, parse the agent's intermediate steps
        contexts.append(["retrieved document chunk"])

    result = evaluate_rag(
        questions,
        answers,
        contexts,
        ground_truths
    )

    print(result)


if __name__ == "__main__":
    run_evaluation()
