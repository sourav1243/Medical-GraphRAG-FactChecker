# Copyright 2025 Sourav Kumar Sharma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test suite for search functionality.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        from src.search import cosine_similarity

        A = np.array([1.0, 0.0, 0.0])
        B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        scores = cosine_similarity(A, B)

        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        from src.search import cosine_similarity

        A = np.array([1.0, 0.0])
        B = np.array([[-1.0, 0.0]])

        scores = cosine_similarity(A, B)

        assert scores[0] == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        from src.search import cosine_similarity

        A = np.array([1.0, 0.0])
        B = np.array([[0.0, 1.0]])

        scores = cosine_similarity(A, B)

        assert scores[0] == pytest.approx(0.0)

    def test_multiple_vectors(self):
        """Test cosine similarity with multiple corpus vectors."""
        from src.search import cosine_similarity

        A = np.array([1.0, 0.0, 0.0])
        B = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.707, 0.707, 0.0],
        ])

        scores = cosine_similarity(A, B)

        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(0.707, abs=0.01)


class TestSearchPubmedNaive:
    """Tests for search_pubmed_naive function."""

    @patch("src.search.get_model")
    @patch("src.search.load_embeddings")
    def test_search_returns_list(self, mock_load, mock_get_model):
        """Test that search returns a list of results."""
        mock_embeddings = np.random.rand(10, 1024).astype(np.float32)
        mock_data = [
            {"question": f"Q{i}", "context": f"ctx{i}", "answer": "yes", "label": "yes"}
            for i in range(10)
        ]

        mock_load.return_value = (mock_embeddings, mock_data)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1024).astype(np.float32)
        mock_get_model.return_value = mock_model

        from src.search import search_pubmed_naive

        results = search_pubmed_naive("test claim", top_k=3)

        assert isinstance(results, list)
        assert len(results) == 3
        assert "score" in results[0]
        assert "question" in results[0]

    @patch("src.search.get_model")
    @patch("src.search.load_embeddings")
    def test_search_returns_scores_in_range(self, mock_load, mock_get_model):
        """Test that scores are in valid range."""
        mock_embeddings = np.random.rand(10, 1024).astype(np.float32)
        mock_embeddings = mock_embeddings / np.linalg.norm(
            mock_embeddings, axis=1, keepdims=True
        )

        mock_data = [
            {"question": f"Q{i}", "context": f"ctx{i}", "answer": "yes", "label": "yes"}
            for i in range(10)
        ]

        mock_load.return_value = (mock_embeddings, mock_data)
        mock_model = MagicMock()
        query_emb = mock_embeddings[0].copy()
        mock_model.encode.return_value = query_emb
        mock_get_model.return_value = mock_model

        from src.search import search_pubmed_naive

        results = search_pubmed_naive("test", top_k=5)

        for r in results:
            assert -1.0 <= r["score"] <= 1.001  # Allow small floating-point error


class TestGetModel:
    """Tests for get_model function."""

    @patch("src.search.model", None)
    @patch("src.search.SentenceTransformer")
    def test_get_model_returns_singleton(self, mock_st):
        """Test that get_model returns the same instance."""
        from src.search import get_model

        model1 = get_model()
        model2 = get_model()

        assert model1 is model2

    @patch("src.search.model", None)
    @patch("src.search.SentenceTransformer")
    def test_get_model_loads_correct_name(self, mock_st):
        """Test that correct model is loaded."""
        from src.search import get_model
        import src.search

        src.search.model = None

        get_model()

        mock_st.assert_called_once_with("BAAI/bge-m3")
