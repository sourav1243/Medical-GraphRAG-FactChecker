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
Test suite for Neo4j graph functionality using mocks.
"""

from unittest.mock import MagicMock, patch


class TestGraphQueries:
    """Tests for graph query functions."""

    def test_query_graph_returns_expected_structure(self):
        """Ensure the graph query function returns a list of dicts with required keys."""
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "text": "Ibuprofen treats inflammation.",
            "score": 0.91,
            "metadata": {
                "relationship": "TREATS",
                "neighbor_name": "Inflammation",
                "neighbor_type": "Symptom",
            },
        }[key]

        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([mock_record]))

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.graph.driver", mock_driver):
            from src.graph import query_graph

            results = query_graph("ibuprofen inflammation")

            assert isinstance(results, dict)
            assert "entities" in results
            assert "relations" in results

    def test_get_graph_stats_returns_counts(self):
        """Test that get_graph_stats returns expected structure."""
        mock_driver = MagicMock()

        mock_session = MagicMock()

        result1 = MagicMock()
        result1.single.return_value = {"c": 10}

        result2 = MagicMock()
        result2.single.return_value = {"c": 25}

        result3 = MagicMock()
        result3.single.return_value = {"c": 50}

        mock_session.run.side_effect = [
            result1,
            result2,
            result3,
        ]

        mock_driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("src.graph.driver", mock_driver):
            with patch("src.graph.get_driver", return_value=mock_driver):
                from src.graph import get_graph_stats

                stats = get_graph_stats()

                assert "chunks" in stats
                assert "entities" in stats
                assert "relations" in stats
                assert stats["chunks"] == 10
                assert stats["entities"] == 25
                assert stats["relations"] == 50


class TestGraphDriver:
    """Tests for graph driver management."""

    @patch("src.graph.driver", None)
    @patch("src.graph.GraphDatabase")
    def test_get_driver_creates_driver(self, mock_gd):
        """Test that get_driver creates a new driver."""
        mock_driver = MagicMock()
        mock_gd.driver.return_value = mock_driver

        import src.graph
        src.graph.driver = None

        from src.graph import get_driver

        driver = get_driver()

        mock_gd.driver.assert_called_once()

    @patch("src.graph.driver", None)
    @patch("src.graph.GraphDatabase")
    def test_close_driver_sets_to_none(self, mock_gd):
        """Test that close_driver sets the driver to None."""
        mock_driver = MagicMock()
        mock_gd.driver.return_value = mock_driver

        import src.graph
        src.graph.driver = mock_driver

        from src.graph import close_driver

        close_driver()

        assert src.graph.driver is None


class TestNeo4jConnection:
    """Tests for Neo4j connection handling."""

    def test_driver_uses_correct_uri(self):
        """Test that driver is initialized with correct URI."""
        with patch("src.graph.GraphDatabase") as mock_gd:
            with patch("src.graph.settings") as mock_settings:
                mock_settings.neo4j_uri = "bolt://localhost:7687"
                mock_settings.neo4j_username = "neo4j"
                mock_settings.neo4j_password = "password"

                mock_driver = MagicMock()
                mock_gd.driver.return_value = mock_driver

                import src.graph
                src.graph.driver = None

                from src.graph import get_driver

                get_driver()

                mock_gd.driver.assert_called_with(
                    "bolt://localhost:7687",
                    auth=("neo4j", "password"),
                )
