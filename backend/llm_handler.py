"""
Google Gemini LLM integration module for Reading Passage Evidence Locator.
Uses the new official google-genai SDK (`from google import genai`).
Strictly conditions answer generation on retrieved evidence passages.
Prohibits outside knowledge, ensures sentence-level citations, and
strictly abstains when evidence is insufficient.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_PHRASE = "Insufficient information is available in the uploaded documents to answer this question."


class LLMHandler:
    """Manages Gemini API calls with strict evidence grounding and abstention."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY environment variable.
            model_name: Target Gemini model identifier (default: 'gemini-3.5-flash-lite').
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite")
        self._client: Optional[genai.Client] = None

        if self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to create Google GenAI Client: {e}")
                self._client = None
        else:
            logger.warning("GEMINI_API_KEY is not set. LLM answer generation will require key before querying.")

    def set_api_key(self, api_key: str):
        """Updates the Gemini API key and reinitializes the client."""
        self.api_key = api_key
        try:
            self._client = genai.Client(api_key=self.api_key)
            logger.info("Successfully initialized Google GenAI Client with updated API key.")
        except Exception as e:
            logger.error(f"Failed to reinitialize GenAI client: {e}")
            self._client = None

    def test_connection(self) -> Dict[str, Any]:
        """
        Sends a simple verification prompt to test Gemini connectivity.
        Prompt: 'Reply with exactly: GEMINI_OK'

        Returns:
            Dict: {'status': 'success' | 'error', 'response': str, 'message': str}
        """
        if not self._client:
            current_key = self.api_key or os.getenv("GEMINI_API_KEY")
            if current_key:
                self.set_api_key(current_key)

        if not self._client:
            return {
                "status": "error",
                "response": "",
                "message": "Google Gemini API key is not configured in environment or .env file."
            }

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents="Reply with exactly: GEMINI_OK"
            )
            raw_text = (response.text or "").strip()
            return {
                "status": "success",
                "response": raw_text,
                "message": f"Connection verified successfully with model '{self.model_name}'."
            }
        except APIError as api_err:
            logger.error(f"Gemini API Error during connection test: {api_err}")
            return {
                "status": "error",
                "response": "",
                "message": f"Gemini API Error: {api_err.message}"
            }
        except Exception as e:
            logger.error(f"Error during Gemini connection test: {e}")
            return {
                "status": "error",
                "response": "",
                "message": str(e)
            }

    def generate_answer(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates an evidence-grounded answer with citations or abstains.

        Args:
            query: The user's question.
            evidence_chunks: List of relevant evidence passages from EvidenceRetriever.

        Returns:
            Dict:
            {
                "answer": str,
                "abstained": bool,
                "citations": List[Dict[str, Any]],
                "status": "success" | "insufficient_evidence" | "error",
                "error_message": Optional[str]
            }
        """
        # If no evidence chunks provided, abstain immediately
        if not evidence_chunks:
            return {
                "answer": INSUFFICIENT_EVIDENCE_PHRASE,
                "abstained": True,
                "citations": [],
                "status": "insufficient_evidence",
                "error_message": None
            }

        # Check API client readiness
        if not self._client:
            # Try reloading from env
            current_key = self.api_key or os.getenv("GEMINI_API_KEY")
            if current_key:
                self.set_api_key(current_key)

        if not self._client:
            return {
                "answer": "Error: Google Gemini API key is not configured. Please add GEMINI_API_KEY to your .env file.",
                "abstained": False,
                "citations": [],
                "status": "error",
                "error_message": "Missing GEMINI_API_KEY"
            }

        prompt = self._build_prompt(query, evidence_chunks)

        try:
            config = types.GenerateContentConfig(
                temperature=0.0,  # Zero temperature for deterministic, factual outputs
                top_p=0.9,
                max_output_tokens=1024
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            raw_answer = (response.text or "").strip()

            # Check for abstention
            if self._is_abstention(raw_answer):
                return {
                    "answer": INSUFFICIENT_EVIDENCE_PHRASE,
                    "abstained": True,
                    "citations": [],
                    "status": "insufficient_evidence",
                    "error_message": None
                }

            # Map citations
            citations = self._extract_citations(evidence_chunks)

            return {
                "answer": raw_answer,
                "abstained": False,
                "citations": citations,
                "status": "success",
                "error_message": None
            }

        except APIError as api_err:
            logger.error(f"Gemini API Error: {api_err}")
            return {
                "answer": f"Gemini API Error: {api_err.message}",
                "abstained": False,
                "citations": [],
                "status": "error",
                "error_message": str(api_err)
            }
        except Exception as e:
            logger.error(f"Unexpected error during answer generation: {e}")
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "abstained": False,
                "citations": [],
                "status": "error",
                "error_message": str(e)
            }

    def _build_prompt(self, query: str, evidence_chunks: List[Dict[str, Any]]) -> str:
        """Constructs a strict evidence-conditioned prompt."""
        context_blocks = []
        for i, chunk in enumerate(evidence_chunks, start=1):
            doc = chunk.get("document_name", "Unknown Document")
            page = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            context_blocks.append(
                f"[Source {i}] Document: {doc} | Page: {page}\nExcerpt: \"{text}\""
            )

        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are the Reading Passage Evidence Locator (PS 77).
Your mission is to provide strictly factual, evidence-grounded answers based EXCLUSIVELY on the provided source excerpts.

### MANDATORY INSTRUCTIONS:
1. Answer the question using ONLY the factual statements directly present in the source excerpts below.
2. DO NOT use any outside knowledge, assumptions, or personal speculation.
3. Every factual claim in your answer MUST cite its source document and page number in brackets, for example: [Document: filename.pdf, Page: 2].
4. If the source excerpts DO NOT contain sufficient evidence or direct facts to answer the question, you MUST reply EXACTLY with:
"{INSUFFICIENT_EVIDENCE_PHRASE}"
5. Never fabricate, guess, or invent citations.

### SOURCE EXCERPTS:
{context_str}

### QUESTION:
{query}

### VERIFIED EVIDENCE ANSWER:"""
        return prompt

    def _is_abstention(self, text: str) -> bool:
        """Determines if the LLM output signifies abstention."""
        text_lower = text.lower()
        key_phrases = [
            INSUFFICIENT_EVIDENCE_PHRASE.lower(),
            "insufficient information",
            "not enough evidence",
            "does not contain sufficient evidence",
            "not mentioned in the provided",
            "the provided excerpts do not contain",
            "the uploaded documents do not contain"
        ]
        return any(phrase in text_lower for phrase in key_phrases)

    @staticmethod
    def _extract_citations(evidence_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compiles clean citation list from the supporting chunks."""
        citations = []
        for chunk in evidence_chunks:
            citations.append({
                "document_name": chunk.get("document_name"),
                "page_number": chunk.get("page_number"),
                "supporting_text": chunk.get("text"),
                "similarity_score": chunk.get("similarity_score")
            })
        return citations
