"""Evaluación focalizada de faithfulness y relevancia en respuestas RAG."""

from __future__ import annotations

from typing import Any


def evaluate_response(question: str, response: Any, llm: Any) -> dict[str, Any]:
    """Evalúa una respuesta LlamaIndex usando sus nodos fuente como contexto."""
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

    faithfulness = FaithfulnessEvaluator(llm=llm).evaluate_response(response=response)
    relevancy = RelevancyEvaluator(llm=llm).evaluate_response(
        query=question,
        response=response,
    )
    return {
        "faithfulness": {
            "score": faithfulness.score,
            "passing": faithfulness.passing,
            "feedback": faithfulness.feedback,
        },
        "relevancy": {
            "score": relevancy.score,
            "passing": relevancy.passing,
            "feedback": relevancy.feedback,
        },
    }
