from typing import Tuple, List

from sklearn.model_selection import StratifiedKFold
from transformers import BertModel, BertTokenizer
import torch


class RatingPredictionModel(torch.nn.Module):

    def __init__(self):
        super(RatingPredictionModel, self).__init__()
        self.dense_stack = torch.nn.Sequential(
            torch.nn.Linear(768, 1)
        )

    def forward(self, input):
        return torch.exp(self.dense_stack(input))


class ConsistencyPredictionModel(torch.nn.Module):

    def __init__(self):
        super(ConsistencyPredictionModel, self).__init__()
        pass

    def forward(self, input):
        pass


def get_bert_sentence_representation(sentences: list, bert, tokenizer):
    return bert(**tokenizer(sentences, return_tensors="pt", padding=True)).pooler_output.detach()


def load_data(sentence_path: str, rating_path: str) -> Tuple[List[str], torch.Tensor]:
    sentences = read_file_with_data(sentence_path)
    ratings = torch.Tensor([int(rating) for rating in read_file_with_data(rating_path)])
    return sentences, ratings


def read_file_with_data(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = f.readlines()
    return lines


def cross_validate(X, y, metrics, epoch_num=50):
    split_generator = StratifiedKFold(n_splits=10, shuffle=True)
    splits = split_generator.split(X, y)
    fold_results = []
    for i, (train_ind, test_ind) in enumerate(splits):
        print(f"\n{i + 1}-fold:")
        X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]
        model, optimizer, loss_function = init_model()
        train(model, optimizer, loss_function, X_train, y_train, epoch_num)
        evaluation_results = evaluate(model, metrics, X_test, y_test)
        print(f"Test set variance: {torch.var(y_test)}")
        for metric, results in zip(metrics, evaluation_results):
            print(f"{metric}: {results}")
        fold_results.append(evaluation_results)
    averaged_results_over_fold = torch.mean(torch.Tensor(fold_results), 0)
    print("\nResults averaged over folds:")
    print(f"Variance: {torch.var(y)}")
    for metric, results in zip(metrics, averaged_results_over_fold):
        print(f"{metric}: {results}")
    print()


def init_model():
    model = RatingPredictionModel()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=L2_REGULARIZATION_WEIGHT)
    loss_function = torch.nn.MSELoss()
    return model, optimizer, loss_function


def train(model, optimizer, loss_function, X, y, n_epochs=50, verbose=False):
    for epoch in range(n_epochs):
        loss_sum = torch.tensor(0.)
        for i in range(0, len(y), BATCH_SIZE):
            pred = model(X[i:i + BATCH_SIZE]).squeeze()
            loss = loss_function(pred, torch.Tensor(y[i:i + BATCH_SIZE]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad()
        loss_sum += loss
        if verbose:
            print(f"Epoch {epoch + 1} loss: {loss_sum / len(y)}")
    if not verbose:
        print(f"Loss: {loss_sum / len(y)}")


def evaluate(model, metrics, X, y):
    y_pred = infer(model, X)
    return [metric(y, y_pred) for metric in metrics]


def infer(model, X):
    with torch.no_grad():
        prediction = model(X).squeeze()
    return prediction


BATCH_SIZE = 20
L2_REGULARIZATION_WEIGHT = 0.5
PRETRAINED_MODEL_NAME = "bert-base-uncased"

PROMPT_TEXTS = {
    1: "An example of a very negative review is",
    2: "An example of a slightly negative review is",
    3: "An example of a neutral/mixed review is",
    4: "An example of a slightly positive review is",
    5: "An example of a very positive review is"
}


def main():
    sentences, ratings = load_data("../models/Yelp/gt explanations sample.txt", "../models/Yelp/gt sample human labels.txt")
    test_sentences = ["the brunch here is worth a return trip", "i expect more than a tiny cup of juice",
                        "this place has consistent food quality", "i have mixed feelings about this place",
                        "i got two free tickets and i still regret going",
                        "absolutely bad and terrible, worst I have ever eaten", "pretty much average", "fantastic, great, the best i've ever visited"]

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)

    model_input_features = get_bert_sentence_representation(sentences, bert, tokenizer)

    model, optimizer, loss_function = init_model()

    cross_validate(model_input_features, ratings, [loss_function, torch.nn.L1Loss()])

    train(model, optimizer, loss_function, model_input_features, ratings)
    print("Predictions:", infer(model, get_bert_sentence_representation(test_sentences, bert, tokenizer)))


if __name__ == "__main__":
    main()

