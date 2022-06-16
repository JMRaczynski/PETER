import torch
import time
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig
from scipy.stats import spearmanr

from utils import DataLoader


def load_test_data_from_pickle(pickle_path, index_path):
    corpus = DataLoader(pickle_path, index_path, 20000)
    return [i["text"] for i in corpus.test]


def load_test_data_from_txt(txt_path):
    with open(txt_path, "r") as file:
        lines = file.readlines()
    gt = [extract_explanation_and_rating(lines[i]) for i in range(0, len(lines), 4)]
    predictions = [extract_explanation_and_rating(lines[i]) for i in range(2, len(lines), 4)]
    return gt, predictions


def extract_explanation_and_rating(line):
    return line[:line.rindex(" ")], line[line.rindex(" ") + 1:len(line) - 1]


def postprocess_results(results):
    return [result["score"] if result["label"] == "Positive" else 1 - result["score"] for result in results]


PICKLE_PATH = "../../../datasets/Amazon/ClothingShoesAndJewelry/reviews.pickle"
INDEX_PATH = "../../../datasets/Amazon/ClothingShoesAndJewelry/1"

TXT_PATH = "../models/Yelp/generatedyelp.txt"

BATCH_SIZE = 128

device = torch.device("cuda")
cpu = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest").to(device)

label_dict = {0: "Negative", 1: "Neutral", 2: "Positive"}


#sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)
start = time.time()
for x in [load_test_data_from_txt(TXT_PATH)[1]]:
    results = []
    texts, ratings = [tupel[0] for tupel in x], [float(tupel[1]) for tupel in x]
    for i in range(0, len(texts), BATCH_SIZE):
        encoded_input = tokenizer(texts[i:i + BATCH_SIZE], return_tensors='pt', padding=True).to(device)
        results.append(torch.nn.functional.softmax(model(**encoded_input).logits, dim=1).detach().to(cpu))
    concatenated_result = torch.cat(results, dim=0)
    sentiment_ratings = (1 * concatenated_result[:, 0] + 3 * concatenated_result[:, 1] + 5 * concatenated_result[:, 2])
    #processed_results = postprocess_results(results)
    print("PEARSON:", np.corrcoef(ratings[:len(sentiment_ratings)], sentiment_ratings))
    print("Spearman:", spearmanr(ratings[:len(sentiment_ratings)], sentiment_ratings)[0])
# result = sentiment_pipeline(data[:5])
# print(result)
print(time.time() - start)

#data = load_test_data_from_pickle(PICKLE_PATH, INDEX_PATH)
