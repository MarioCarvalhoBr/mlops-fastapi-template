from src.models.dummy_model import DummyModel


class TestDummyModel:
    """Test class for DummyModel."""

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        model = DummyModel()
        assert model.model_name == "dummy_classifier"
        assert model.version == "1.0.0"

    def test_predict_with_good_word(self):
        """Test that the model returns positive for 'good'."""
        model = DummyModel()
        result = model.predict("This is good")
        assert result == "positive"

    def test_predict_with_great_word(self):
        """Test that the model returns positive for 'great'."""
        model = DummyModel()
        result = model.predict("This is great")
        assert result == "positive"

    def test_predict_without_keywords(self):
        """Test that the model returns negative without keywords."""
        model = DummyModel()
        test_inputs = ["test", "random text", "negative sentiment"]
        for input_text in test_inputs:
            result = model.predict(input_text)
            assert result == "negative"
