import sacrebleu as scb
from packaging import version

from rouge_score import rouge_scorer, scoring
import json
import argparse

import sacrebleu as scb
from packaging import version
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def b2_eval(ground_truth: list, predictions: list):
    if len(ground_truth) != len(predictions):
        raise ValueError("Sentence bleu requires the same number of references for each prediction")

    ground_truth_list = [tmp.split(' ') for tmp in ground_truth]
    predictions_list = [tmp.split(' ') for tmp in predictions]
    b2_res_list = [
        sentence_bleu([reference], hypothesis, weights=[0.5, 0.5], smoothing_function=SmoothingFunction().method3) for
        reference, hypothesis in zip(ground_truth_list, predictions_list)]
    b2_res = sum(b2_res_list) / len(b2_res_list)
    return b2_res * 100


class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


class RougeEvaluator:
    def __init__(self, rouge_types=None, use_aggregator=True, use_stemmer=False, tokenizer=None):
        if rouge_types is None:
            rouge_types = ["rougeL"]
        self.rouge_types = rouge_types
        self.use_aggregator = use_aggregator
        self.use_stemmer = use_stemmer
        self.tokenizer = Tokenizer(tokenizer) if tokenizer is not None else None

        self.scorer = rouge_scorer.RougeScorer(rouge_types=self.rouge_types,
                                               use_stemmer=self.use_stemmer,
                                               tokenizer=self.tokenizer)
        if self.use_aggregator:
            self.aggregator = scoring.BootstrapAggregator()
        else:
            self.scores = []

    def is_num_equal(self, references, predictions):
        if len(references) != len(predictions):
            return {'error': 'preds and refrs have different length'}
        else:
            return True

    def compute_rouge_l(self, references, predictions):
        multi_ref = isinstance(references[0], list)

        if self.is_num_equal(references, predictions) != True:
            return self.is_num_equal(references, predictions)

        for ref, pred in zip(references, predictions):
            if multi_ref:
                score = self.scorer.score_multi(ref, pred)
            else:
                score = self.scorer.score(ref, pred)
            if self.use_aggregator:
                self.aggregator.add_scores(score)
            else:
                self.scores.append(score)

        if self.use_aggregator:
            result = self.aggregator.aggregate()
            for key in result:
                result[key] = result[key].mid.fmeasure
        else:
            result = {}
            for key in self.scores[0]:
                result[key] = list(score[key].fmeasure for score in self.scores)

        return result

    def rl_eval(ground_truth: list, predictions: list):
        rl_eval_obj = RougeEvaluator()
        rl_res = rl_eval_obj.compute_rouge_l(references=ground_truth, predictions=predictions)
        return rl_res


    def rl_eval_file(ground_truth_path: str, predictions_path: str):
        with open(ground_truth_path, 'r', encoding='utf-8') as ground_truth_f, open(predictions_path, 'r',
                                                                                    encoding='utf-8') as predictions_f:
            ground_truth = ground_truth_f.readlines()
            predictions = predictions_f.readlines()
            rl_eval_res = rl_eval(ground_truth, predictions)
            print(f"rl_eval_res:{rl_eval_res}")

    def b2_eval_file(ground_truth_path: str, predictions_path: str):
        with open(ground_truth_path, 'r', encoding='utf-8') as ground_truth_f, open(predictions_path, 'r',
                                                                                    encoding='utf-8') as predictions_f:
            ground_truth = ground_truth_f.readlines()
            predictions = predictions_f.readlines()
            b2_eval_res = b2_eval(ground_truth, predictions)
            print(f"b2_eval_res:{b2_eval_res}")

    def eval(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = file.readlines()
        reference = []
        predict = []
        for meta_data in data:
            message = json.loads(meta_data)
            predict.append(message['messages'][1]['content'])
            reference.append(message['messages'][1]['old_content'])
        rl_eval = RougeEvaluator()
        # b2_eval = BleuEvaluator()
        rl_res = rl_eval.compute_rouge_l(references=reference, predictions=predict)
        # b2_res = b2_eval.compute_bleu_2(references=reference, predictions=predict)
        b2_res = b2_eval(ground_truth=reference, predictions=predict)
        print('Rl score : ', rl_res['rougeL'])
        print('B2 score : ', b2_res)

    def eval_shac(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = file.readlines()

        reference = []
        predict = []

        for line in data:
            message = json.loads(line)
            predict.append(message['content'])
            reference.append(message['old_content'])

        rl_eval = RougeEvaluator()
        # b2_eval = BleuEvaluator()

        rl_res = rl_eval.compute_rouge_l(references=reference, predictions=predict)
        # b2_res = b2_eval.compute_bleu_2(references=reference, predictions=predict)
        b2_res = b2_eval(ground_truth=reference, predictions=predict)
        print('Rl score : ', rl_res['rougeL'])
        print('B2 score : ', b2_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval GYAFC')
    parser.add_argument('--file', type=str, required=True, help='Path to the input JSON file')
    args = parser.parse_args()
    eval(args.file)
    # eval_shac(args.file)
