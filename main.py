from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis",
                      model="blanchefort/rubert-base-cased-sentiment")


@app.get("/")
def root():
    return {"message": "Демонстрационное ИИ приложение для курса DevOps"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
