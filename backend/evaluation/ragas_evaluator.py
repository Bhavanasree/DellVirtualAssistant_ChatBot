
from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from langchain_aws import ChatBedrock
from ragas.llms import LangchainLLMWrapper

from langchain_aws import BedrockEmbeddings

# Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1"
)

bedrock_llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="ap-south-1"
)

# Wrap for RAGAS
ragas_llm = LangchainLLMWrapper(bedrock_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)

def evaluate_rag(questions, answers, contexts, ground_truths):

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    return result
