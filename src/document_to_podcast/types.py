from pydantic import BaseModel

class SpeechParams(BaseModel):
    speed: float = 1.0
    volume: float = 1.0 