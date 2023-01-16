import argparse
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


def load_data_from_tsv(data_path: str) -> Tuple[List[str], np.ndarray, torch.Tensor]:
    data = np.loadtxt(data_path, dtype=object, delimiter='\t')
    sentences = list(data[:, 0])
    ratings = data[:, 1].astype(np.int32)
    consistency_labels = torch.LongTensor(data[:, 2].astype(np.int64)).to(DEVICE)
    return sentences, ratings, consistency_labels


def load_peter_output(path: str, mod=2) -> Tuple[List[str], List[int]]:
    with open(path, "r") as f:
        lines = f.readlines()
    split_lines = [line.split(" ") for i, line in enumerate(lines) if i % 4 == mod]
    predicted_ratings = [round(float(line[-1])) for line in split_lines]
    return [combine_rating_with_explanation(rating, " ".join(line[:-1]))
            for rating, line in zip(predicted_ratings, split_lines)], predicted_ratings


def combine_rating_with_explanation(rating: int, explanation: str) -> str:
    return PROMPT_TEXTS[rating] + '"' + explanation + '"'


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
    per_class_scores = compute_per_class_consistencies(consistency_predictions, predicted_ratings)
    print(f'Scores per predicted rating: {per_class_scores}')
    return construct_result_np_array(per_class_scores, 100 * number_of_consistent_samples.item() / len(test_sentences))


def construct_result_np_array(per_class_results: dict, full_result: float):
    result_list = [0] * 5
    for rating, score in per_class_results.items():
        result_list[rating - 1] = score
    return np.array(result_list + [full_result])


def compute_per_class_consistencies(consistency_predictions: list, predicted_ratings: list) -> dict:
    results_per_class = dict.fromkeys(range(1, 6))
    for key in results_per_class:
        boolean_mask = np.array(predicted_ratings) == key
        results_per_class[key] = round(100 * np.sum(np.array(consistency_predictions)[boolean_mask]) / np.sum(boolean_mask), 2)
    return results_per_class


def infer(model, X):
    with torch.no_grad():
        prediction = model(X).squeeze()
    return prediction


def evaluation_model_sanity_check(model, test_sentences, bert, tokenizer):
    predictions = torch.argmax(infer(model, get_bert_sentence_representation(test_sentences, bert, tokenizer)), dim=1)
    padding_size = max(len(sentence) for sentence in test_sentences)  # for pretty print :)
    for i, sentence in enumerate(test_sentences):
        print(f'{sentence.ljust(padding_size)}\tPredicted consistency: {predictions[i]}'
              f'\tExpected consistency: {EXPECTED_CONSISTENCIES[i]}')


################################## GLOBAL PARAMETERS DEFINITIONS ###############################


parser = argparse.ArgumentParser(description='Explanation coherence automatic evaluator for PETER')
parser.add_argument('--dataset', type=str, default=None,
                    help='dataset name', choices=['AmazonMovies', 'TripAdvisor', 'Yelp'])
parser.add_argument('--split_number', type=int, default=1,
                    help='dataset split number', choices=[1, 2, 3, 4, 5])
parser.add_argument('--training_set_size', type=int, default=100,
                    help='size of training dataset, used to infer dataset filename')
args = parser.parse_args()

PRETRAINED_MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda")
BATCH_SIZE = 10
INFERENCE_BATCH_SIZE = 64

DATASET = args.dataset
SPLIT = args.split_number
TRAINING_SET_SIZE = args.training_set_size

if DATASET == "TripAdvisor":
    CLASS_WEIGHTS = torch.Tensor([2., 1.])
    L2_REGULARIZATION_WEIGHT = 0.1
    EPOCH_NUM = 150
elif DATASET == "Yelp":
    CLASS_WEIGHTS = torch.Tensor([2.5, 1.])
    L2_REGULARIZATION_WEIGHT = 0.05
    EPOCH_NUM = 150
elif DATASET == "AmazonMovies":
    CLASS_WEIGHTS = torch.Tensor([1.8, 1.])
    L2_REGULARIZATION_WEIGHT = 0.05
    EPOCH_NUM = 100


CONSISTENCY_DATASET_FILE_PATH = f"../models/{DATASET}/gt_explanations_with_labels_{TRAINING_SET_SIZE}samples.tsv"
PETER_OUTPUT_BASE_PATH = f"../models/{DATASET}/{SPLIT}/"

EVALUATED_PETERS = ['PETERplus', 'PETERRplus']

CONSISTENCY_MODEL_DIR = f"consistency_models/{DATASET}/100_samples"
LOAD_CONSISTENCY_MODEL = True

CLASSIFICATION_METRICS = [accuracy_score, recall_score, f1_score, precision_score]

NUMBER_OF_EVALUATION_MODELS = 10

PROMPT_TEXTS = {
    1: "An example of a very negative review is ",
    2: "An example of a slightly negative review is ",
    3: "An example of a neutral or mixed review is ",
    4: "An example of a slightly positive review is ",
    5: "An example of a very positive review is "
}

SANITY_CHECK_SENTENCES = ["the brunch here is worth a return trip", "i expect more than a tiny cup of juice",
                          "this place has consistent food quality", "i have mixed feelings about this place",
                          "i got two free tickets and i still regret going",
                          "terrible service and rooms", "average hotel",
                          "fantastic, great, the best i've ever visited and friendly service"]
SANITY_CHECK_RATINGS = [4, 2, 5, 3, 1, 5, 3, 1]
EXPECTED_CONSISTENCIES = [1, 1, 1, 1, 1, 0, 1, 0]

######################### END OF GLOBAL PARAMETERS DEFINITIONS ################################


def main():
    print(f"Dataset {DATASET} split {SPLIT}")
    sentences, ratings, consistency_labels = load_data_from_tsv(CONSISTENCY_DATASET_FILE_PATH)
    sentences = [combine_rating_with_explanation(rating, sentence)
                      for sentence, rating in zip(sentences, ratings)]
    test_sentences = [combine_rating_with_explanation(rating, sentence)
                      for sentence, rating in zip(SANITY_CHECK_SENTENCES, SANITY_CHECK_RATINGS)]

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME).to(DEVICE)

    model_input_features = torch.cat([get_bert_sentence_representation(sentences[i:i + 100], bert, tokenizer)
                                      for i in range(0, len(sentences), 100)])

    results = {peter_name: [] for peter_name in EVALUATED_PETERS}
    start = time.time()
    for i in range(NUMBER_OF_EVALUATION_MODELS):
        model, optimizer, loss_function = init_model(ConsistencyPredictionModel)

        # cross_validate(model_input_features, consistency_labels, ConsistencyPredictionModel, CLASSIFICATION_METRICS, EPOCH_NUM)

        checkpoint_path = f"{CONSISTENCY_MODEL_DIR}/model{i + 1}.pt"
        if LOAD_CONSISTENCY_MODEL:
            print(f"Evaluation model number {i + 1} loaded\n")
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"Evaluation model number {i + 1} trained")
            train(model, optimizer, loss_function, model_input_features, consistency_labels, EPOCH_NUM)
            torch.save(model.state_dict(), checkpoint_path)
        model.eval()

        # evaluation_model_sanity_check(model, test_sentences, bert, tokenizer)

        # print(evaluate(model, CLASSIFICATION_METRICS, model_input_features, consistency_labels))

        for peter_type in EVALUATED_PETERS:
            peter_output_path = f'{PETER_OUTPUT_BASE_PATH}generated_{DATASET.lower()}_{peter_type}.txt'
            results[peter_type].append(evaluate_on_full_dataset(model, bert, tokenizer, peter_output_path))
    for peter_type in EVALUATED_PETERS:
        results_array = np.c_[tuple(results[peter_type])]
        results_array_with_mean = np.c_[results_array, results_array.mean(axis=1)]
        print(f'{peter_type} consistency:', results_array_with_mean)
        np.savetxt(f'{PETER_OUTPUT_BASE_PATH}{peter_type}_coherence_automatic_eval_results.tsv',
                   results_array_with_mean, fmt="%.2f", header=f'{DATASET} split {SPLIT} {peter_type}', delimiter='\t')
    print(f"\nElapsed time: {time.time() - start} seconds")


if __name__ == "__main__":
    main()

# Amazon Movies 100 epochs, 0.15L2, 1.75;1 weights

# TripAdvisor 150 epochs, 0.1L2, 2.;1 weights

# Yelp, TripAdvisor 150 epochs, 0.05L2, 2.5;1 weights