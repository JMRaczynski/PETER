import time
from typing import Tuple, List

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import BertModel, BertTokenizer
import torch


class RatingPredictionModel(torch.nn.Module):

    def __init__(self):
        super(RatingPredictionModel, self).__init__()
        self.dense_stack = torch.nn.Sequential(
            torch.nn.Linear(768, 32),
            torch.nn.Tanh(),
            # torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1)
        )

    def forward(self, input):
        return torch.exp(self.dense_stack(input))


class ConsistencyPredictionModel(torch.nn.Module):

    def __init__(self):
        super(ConsistencyPredictionModel, self).__init__()
        self.dense_stack = torch.nn.Sequential(
            torch.nn.Linear(768, 32),

            torch.nn.Tanh(),
            # torch.nn.Dropout(0.25),
            torch.nn.Linear(32, 2)
        )

    def forward(self, input):
        return self.dense_stack(input)


def get_bert_sentence_representation(sentences: list, bert, tokenizer):
    return bert(**tokenizer(sentences, return_tensors="pt", padding=True).to(DEVICE)).pooler_output.detach()


def load_data(sentence_path: str, rating_path: str) -> Tuple[List[str], torch.Tensor]:
    sentences = read_file_with_data(sentence_path)
    ratings = torch.Tensor([int(rating) for rating in read_file_with_data(rating_path)])
    return sentences, ratings


def load_consistency_data(path: str) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    lines = read_file_with_data(path)
    split_lines = [line.split("\t") for line in lines]
    samples = [(sentence.split(" "), int(label)) for _, sentence, label in split_lines]
    samples = [(" ".join(sentence_with_rating[:-1]), round(float(sentence_with_rating[-1])), label) for sentence_with_rating, label in samples]
    sentences, ratings, labels = [], [], []
    for sentence, rating, consistency_label in samples:
        sentences.append(sentence)
        ratings.append(rating)
        labels.append(consistency_label)
    return sentences, torch.Tensor(ratings), torch.Tensor(labels).type(torch.LongTensor)


def load_peter_output(path: str, mod=2) -> Tuple[List[str], List[int]]:
    with open(path, "r") as f:
        lines = f.readlines()
    split_lines = [line.split(" ") for i, line in enumerate(lines) if i % 4 == mod]
    predicted_ratings = [round(float(line[-1])) for line in split_lines]
    return [PROMPT_TEXTS[rating] + '"' + " ".join(line[:-1]) + '"' for rating, line in zip(predicted_ratings, split_lines)], predicted_ratings


def read_file_with_data(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = f.readlines()
    return lines


def cross_validate(X, y, model_type, metrics, epoch_num=50):
    split_generator = StratifiedKFold(n_splits=10, shuffle=True)
    splits = split_generator.split(X.cpu(), y.cpu())
    fold_results = []
    for i, (train_ind, test_ind) in enumerate(splits):
        print(f"\n{i + 1}-fold:")
        X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]
        model, optimizer, loss_function = init_model(model_type)
        train(model, optimizer, loss_function, X_train, y_train, epoch_num)
        evaluation_results = evaluate(model, metrics, X_test, y_test)
        print(f"Test set variance: {torch.var(y_test.type(torch.FloatTensor))}")
        for metric, results in zip(metrics, evaluation_results):
            print(f"{metric}: {results}")
        fold_results.append(evaluation_results)
    averaged_results_over_fold = torch.mean(torch.Tensor(fold_results), 0)
    print("\nResults averaged over folds:")
    print(f"Variance: {torch.var(y.type(torch.FloatTensor))}")
    for metric, results in zip(metrics, averaged_results_over_fold):
        print(f"{metric}: {results}")
    print()


def init_model(model_type):
    model = model_type().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=L2_REGULARIZATION_WEIGHT)
    loss_function = torch.nn.MSELoss() if model_type == RatingPredictionModel else torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    return model, optimizer, loss_function.to(DEVICE)


def train(model, optimizer, loss_function, X: torch.Tensor, y: torch.Tensor, n_epochs=50, verbose=False):
    batch_num = len(y) / BATCH_SIZE
    for epoch in range(1, n_epochs + 1):
        loss_sum = torch.tensor(0.)
        shuffled_indices = torch.randperm(len(y))
        X, y = X[shuffled_indices], y[shuffled_indices]
        for i in range(0, len(y), BATCH_SIZE):
            pred = model(X[i:i + BATCH_SIZE]).squeeze()
            loss = loss_function(pred, y[i:i + BATCH_SIZE])
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.cpu()
        if verbose or epoch == n_epochs:
            print(f"Epoch {epoch} loss: {loss_sum / batch_num}")


def evaluate(model, metrics, X, y):
    y_pred = infer(model, X)
    if set(metrics) == set(CLASSIFICATION_METRICS):
        y_pred = torch.argmax(y_pred, dim=1)
    # print(y_pred)
    return [metric(y.cpu(), y_pred.cpu()) for metric in metrics]


def evaluate_on_full_dataset(model, bert, tokenizer, data_path, mod=2):
    test_sentences, predicted_ratings = load_peter_output(data_path, mod)
    number_of_consistent_samples = 0
    consistency_predictions = []
    for i in range(0, len(test_sentences), INFERENCE_BATCH_SIZE):
        inference_results = infer(model, get_bert_sentence_representation(test_sentences[i:i + INFERENCE_BATCH_SIZE], bert, tokenizer))
        predicted_consistencies = torch.argmax(inference_results, dim=1)
        number_of_consistent_samples += torch.sum(predicted_consistencies)
        consistency_predictions += list(predicted_consistencies.cpu().numpy())
        # print("Predictions:", infer(model, get_bert_sentence_representation(test_sentences[i:i + INFERENCE_BATCH_SIZE], bert, tokenizer)))
    print(compute_per_class_consistencies(consistency_predictions, predicted_ratings))
    return number_of_consistent_samples.item() / len(test_sentences), consistency_predictions


def compute_per_class_consistencies(consistency_predictions: list, predicted_ratings: list) -> dict:
    results_per_class = dict.fromkeys(range(1, 6))
    for key in results_per_class:
        boolean_mask = np.array(predicted_ratings) == key
        results_per_class[key] = round(100 * np.sum(np.array(consistency_predictions)[boolean_mask]) / np.sum(boolean_mask), 2)
    for i in results_per_class.values():
        print(i)
    return results_per_class


def infer(model, X):
    with torch.no_grad():
        prediction = model(X).squeeze()
    return prediction

PRETRAINED_MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda")
BATCH_SIZE = 10
INFERENCE_BATCH_SIZE = 64

# DATASET = "AmazonMovies"
# CLASS_WEIGHTS = torch.Tensor([1.8, 1.])
# L2_REGULARIZATION_WEIGHT = 0.05
# EPOCH_NUM = 100

# DATASET = "TripAdvisor"
# CLASS_WEIGHTS = torch.Tensor([2., 1.])
# L2_REGULARIZATION_WEIGHT = 0.1
# EPOCH_NUM = 150

DATASET = "Yelp"
CLASS_WEIGHTS = torch.Tensor([2.5, 1.])
L2_REGULARIZATION_WEIGHT = 0.05
EPOCH_NUM = 150


SENTENCE_FILE_PATH = f"../models/{DATASET}/gt explanations sample.txt"
GROUND_TRUTH_RATING_FILE_PATH = f"../models/{DATASET}/gt sample labels.txt"
HUMAN_RATING_FILE_PATH = f"../models/{DATASET}/gt sample human labels.txt"
CONSISTENCY_FILE_PATH = f"../models/{DATASET}/gt sample consistency labels.txt"
ALL_DATA_FILE_PATH = f"../models/{DATASET}/consistency_manual_labeled_dataset.txt"
PETER_OUTPUT_PATH = f"../models/{DATASET}/generated{DATASET.lower()}.txt"
IMPROVED_PETER_OUTPUT_PATH = f"../models/{DATASET}/generated_rating_input_yelp.txt"
CONSISTENCY_MODEL_DIR = f"consistency_models/{DATASET}/100_samples"
LOAD_CONSISTENCY_MODEL = True

CLASSIFICATION_METRICS = [accuracy_score, recall_score, f1_score, precision_score]
REGRESSION_METRICS = [torch.nn.MSELoss, torch.nn.L1Loss]

PROMPT_TEXTS = {
    1: "An example of a very negative review is ",
    2: "An example of a slightly negative review is ",
    3: "An example of a neutral or mixed review is ",
    4: "An example of a slightly positive review is ",
    5: "An example of a very positive review is "
}


# def main():
#     sentences, ratings = load_data(SENTENCE_FILE_PATH, RATING_FILE_PATH)
#     # sentences, ratings, _ = load_consistency_data(ALL_DATA_FILE_PATH)
#     test_sentences = ["the brunch here is worth a return trip", "i expect more than a tiny cup of juice",
#                         "this place has consistent food quality", "i have mixed feelings about this place",
#                         "i got two free tickets and i still regret going",
#                         "absolutely bad and terrible, worst I have ever eaten", "pretty much average", "fantastic, great, the best i've ever visited"]
#
#     tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
#     bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
#
#     model_input_features = get_bert_sentence_representation(sentences, bert, tokenizer)
#
#     model, optimizer, loss_function = init_model(RatingPredictionModel)
#
#     cross_validate(model_input_features, ratings, RatingPredictionModel, [loss_function, torch.nn.L1Loss()], 100)
#
#     train(model, optimizer, loss_function, model_input_features, ratings, 100)
#     print("Predictions:", infer(model, get_bert_sentence_representation(test_sentences, bert, tokenizer)))


def main():
    print(f"Dataset: {DATASET}")
    # sentences, ratings, consistency_labels = load_consistency_data(ALL_DATA_FILE_PATH)
    sentences, ratings = load_data(SENTENCE_FILE_PATH, GROUND_TRUTH_RATING_FILE_PATH)
    test_sentences = ["the brunch here is worth a return trip", "i expect more than a tiny cup of juice",
                      "this place has consistent food quality", "i have mixed feelings about this place",
                      "i got two free tickets and i still regret going",
                      "terrible service and rooms", "average hotel", "fantastic, great, the best i've ever visited and friendly service"]
    sentences = [PROMPT_TEXTS[int(rating.item())] + '"' + sentence + '"' for sentence, rating in zip(sentences, ratings)]
    # print(sentences)
    test_sentences = [PROMPT_TEXTS[rating] + '"' + sentence + '"' for sentence, rating in zip(test_sentences, [4, 2, 5, 3, 1, 5, 3, 1])]
    # test_sentences, _ = load_peter_output(PETER_OUTPUT_PATH)
    # print(test_sentences)
    consistency_labels = torch.Tensor([int(is_consistent) for is_consistent in read_file_with_data(CONSISTENCY_FILE_PATH)]).type(torch.LongTensor).to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME).to(DEVICE)

    model_input_features = get_bert_sentence_representation(sentences, bert, tokenizer)

    results_basic, results_improved = [], []
    start = time.time()
    for i in range(10):
        model, optimizer, loss_function = init_model(ConsistencyPredictionModel)

        # cross_validate(model_input_features, consistency_labels, ConsistencyPredictionModel, CLASSIFICATION_METRICS, EPOCH_NUM)

        checkpoint_path = f"{CONSISTENCY_MODEL_DIR}/model{i + 1}.pt"
        if LOAD_CONSISTENCY_MODEL:
            print(f"{i + 1} loaded")
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"{i + 1} trained")
            train(model, optimizer, loss_function, model_input_features, consistency_labels, EPOCH_NUM)
            torch.save(model.state_dict(), checkpoint_path)
        model.eval()
        # print(infer(model, get_bert_sentence_representation(test_sentences, bert, tokenizer)))
        # print(evaluate(model, CLASSIFICATION_METRICS, model_input_features, consistency_labels))
        results, predictions = evaluate_on_full_dataset(model, bert, tokenizer, PETER_OUTPUT_PATH)
        results_basic.append(results)
        results_modification, predictions_modification = evaluate_on_full_dataset(model, bert, tokenizer, IMPROVED_PETER_OUTPUT_PATH)
        results_improved.append(results_modification)
    print(f"Elapsed time: {time.time() - start}")
    print("Basic PETER consistency:", [round(i * 100, 2) for i in results_basic], np.mean(results_basic))
    print("Improved PETER consistency:", [round(i * 100, 2) for i in results_improved], np.mean(results_improved))


if __name__ == "__main__":
    main()

# Amazon Movies 100 epochs, 0.15L2, 1.75;1 weights

# TripAdvisor 150 epochs, 0.1L2, 2.;1 weights

# Yelp, TripAdvisor 150 epochs, 0.05L2, 2.5;1 weights