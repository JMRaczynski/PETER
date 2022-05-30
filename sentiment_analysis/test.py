from transformers import pipeline

from utils import DataLoader

sentiment_pipeline = pipeline("sentiment-analysis")
corpus = DataLoader("../../../datasets/Amazon/ClothingShoesAndJewelry/reviews.pickle", "../../../datasets/Amazon/ClothingShoesAndJewelry/1", 20000)
data = [i["text"] for i in corpus.test]
print(data[:10])
result = sentiment_pipeline(data[:5])
print(result)