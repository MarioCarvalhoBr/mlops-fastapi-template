from src.core.logger import logger
from src.models.dummy_model import DummyModel

model = DummyModel()
logger.info(f"Loaded model: {model.model_name} (v{model.version})")


def predict_service(input_text: str) -> str:
    logger.info(f"Making prediction for input: {input_text[:50]}...")

    try:
        prediction = model.predict(input_text)
        logger.info(f"Prediction result: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
