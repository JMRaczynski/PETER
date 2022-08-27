import torch
import time
import numpy as np
from matplotlib import pyplot as plt
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


def round_to_three_numbers(arr):
    result = np.copy(arr)
    result[result < 2.5] = 1
    result[result > 3.5] = 5
    result[(result > 2) & (result < 4)] = 3
    return result


PICKLE_PATH = "../../../datasets/Amazon/ClothingShoesAndJewelry/reviews.pickle"
INDEX_PATH = "../../../datasets/Amazon/ClothingShoesAndJewelry/1"

TXT_PATH = "../models/Yelp/generatedyelp.txt"

BATCH_SIZE = 64

gpu = torch.device("cuda")
cpu = torch.device("cpu")

# tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
# model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest").to(gpu)

#tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
#model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment").to(gpu)

tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english").to(gpu)

label_dict = {0: "Negative", 1: "Neutral", 2: "Positive"}
lines = [3925, 4693, 7149, 9078, 11035, 12562, 14346, 34326, 34981, 37277, 39921, 44606, 44658, 46009, 48642, 69129, 82865, 91174, 97331, 114029, 117077, 119503, 121053, 124225, 124777, 125803, 127114, 128113, 133939, 136942, 149245, 153041, 155530, 157689, 158262, 163897, 178349, 181025, 181745, 184153, 185062, 188695, 189814, 199450, 201367, 204070, 206341, 209345, 214221, 218153, 220513, 224401, 232221, 245097, 249433, 250187, 253623, 262353, 263265, 272563, 277686, 284121, 296174, 296669, 296686, 305259, 305929, 314570, 335459, 339765, 343515, 346353, 359643, 366238, 369143, 375611, 375915, 379179, 380106, 393315, 398398, 402610, 407854, 411783, 415862, 418957, 422517, 428837, 431257, 444805, 447517, 454359, 456226, 457893, 459962, 466393, 466945, 476899, 501217, 516165]
lines = np.array(lines) // 4

#sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)
start = time.time()
for x in [load_test_data_from_txt(TXT_PATH)[0]]:
    results = []
    texts, ratings = [tupel[0] for tupel in x], torch.Tensor([float(tupel[1]) for tupel in x])
    print(torch.mean((ratings - torch.mean(ratings)) ** 2), torch.var(ratings))
    exit(0)
    for i in range(0, len(texts), BATCH_SIZE):
        encoded_input = tokenizer(texts[i:i + BATCH_SIZE], return_tensors='pt', padding=True).to(gpu)
        results.append(torch.nn.functional.softmax(model(**encoded_input).logits, dim=1).detach().to(cpu))
    concatenated_result = torch.cat(results, dim=0)
    #sentiment_ratings = (1 * concatenated_result[:, 0] + 3 * concatenated_result[:, 1] + 5 * concatenated_result[:, 2])
    #sentiment_ratings = torch.sum(concatenated_result * torch.Tensor([1, 2, 3, 4, 5]), dim=1)
    # sentiment_ratings = round_to_three_numbers(sentiment_ratings)
    # ratings = round_to_three_numbers(np.array(ratings))
    sentiment_ratings = concatenated_result[:, 1] * 4 + 1
    # for i in range(100):
    #     print(sentiment_ratings[i], texts[i])
    print("PEARSON:", np.corrcoef(ratings[:len(sentiment_ratings)], sentiment_ratings))
    print("Spearman:", spearmanr(ratings[:len(sentiment_ratings)], sentiment_ratings)[0])
    print("PEARSON:", np.corrcoef(ratings[lines], sentiment_ratings[lines]))
    print("Spearman:", spearmanr(ratings[lines], sentiment_ratings[lines])[0])
    print("MSE:", torch.nn.functional.mse_loss(ratings[:len(sentiment_ratings)], sentiment_ratings))
    print("MAE:", torch.nn.functional.l1_loss(ratings[:len(sentiment_ratings)], sentiment_ratings))
    plt.scatter(ratings[lines], sentiment_ratings[lines], s=3)
    for i in lines:
        print(texts[i], ratings[i], sentiment_ratings[i])
    plt.show()
# result = sentiment_pipeline(data[:5])
# print(result)
print(time.time() - start)

#data = load_test_data_from_pickle(PICKLE_PATH, INDEX_PATH)
