from typing import Any


class DummyModel:
    """
    Machine Learning model abstraction.
    In a real project, this is where weights and networks are initialized.
    """

    def __init__(self) -> None:
        self.model_name = "dummy_classifier"
        self.version = "1.0.0"

    def predict(self, input_data: Any) -> str:
        text = str(input_data).lower()
        if "good" in text or "great" in text:
            return "positive"
        return "negative"
