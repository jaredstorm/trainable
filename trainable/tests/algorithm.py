from unittest import TestCase

import torch
import random
from trainable.algorithm import Algorithm, AlgorithmList


class BasicTest(TestCase):
    def test(self):
        alg = TestAlgorithmA()
        alg.eval()
        metrics = alg()

        self.assertTrue(
            len(metrics) == 3,
            f"len(metrics) was {len(metrics)} but should have been 3."
        )

        for key in metrics:
            self.assertTrue(
                type(metrics[key]) is float,
                "metrics types incorrect. expected: float, actual: {type(metrics[key])}"
            )

class BasicCompositionTest(TestCase):
    def test(self):
        alg = AlgorithmList(TestAlgorithmA(), TestAlgorithmB())
        alg.eval()
        metrics = alg()

        self.assertTrue(
            len(metrics) == 5,
            f"len(metrics) was {len(metrics)} but should have been 5."
        )

        for key in metrics:
            self.assertTrue(
                type(metrics[key]) is float,
                "metrics types incorrect. expected: float, actual: {type(metrics[key])}"
            )

class LayeredCompostionTest(TestCase):
    def test(self):
        alg1 = AlgorithmList(TestAlgorithmA(), TestAlgorithmB())
        alg2 = AlgorithmList(TestAlgorithmC(), TestAlgorithmD())
        alg = AlgorithmList(alg1, alg2)

        alg.eval()
        metrics = alg()

        self.assertTrue(
            len(metrics) == 7,
            f"len(metrics) was {len(metrics)} but should have been 5."
        )

        for key in metrics:
            self.assertTrue(
                type(metrics[key]) is float,
                "metrics types incorrect. expected: float, actual: {type(metrics[key])}"
            )

class HookTest(TestCase):
    def test(self):
        lst = []
        alg = TestAlgorithmA()

        alg.register_pre_hook(prehook(lst))
        alg.register_post_hook(posthook(lst))
        alg.eval()

        metrics = alg()

        pre = "Pre Call Hook"
        post = "Post Call Hook"
        self.assertTrue(pre in lst, "Pre call hook not run")
        self.assertTrue(lst[0] == pre, "Pre call hook not in right order")
        self.assertTrue(post in lst, "Post call hook not run")
        self.assertTrue(lst[1], "Post call hook not in right order")

class AdvancedHookTest(TestCase):
    def test(self):
        lst = []
        alg = TestAlgorithmA()
        alg.eval()

        alg.register_pre_hook(prehook(lst))
        alg.register_post_hook(posthook(lst))
        alg.register_pre_hook(prehook(lst))
        alg.register_post_hook(posthook(lst))

        metrics = alg()

        pre = "Pre Call Hook"
        post = "Post Call Hook"
        self.assertTrue(pre in lst, "Pre call hook not run")
        self.assertTrue(lst[0] == pre, "Pre call hook not in right order")
        self.assertTrue(lst[1] == pre, "Pre call hook not in right order")

        self.assertTrue(post in lst, "Post call hook not run")
        self.assertTrue(lst[2] == post, "Post call hook not in right order")
        self.assertTrue(lst[3] == post, "Pre call hook not in right order")

class TestEvalTest(TestCase):
    def test(self):
        alg = TestAlgorithmA()
        self.assertFalse(alg._eval, "Algorithm did not start in training mode")

        alg.eval()
        self.assertTrue(alg._eval, "Algorithm did not change to evaluation mode")

        alg.train()
        self.assertFalse(alg._eval, "Algorithm did not change to training mode")

class TestEvalAdvancedTest(TestCase):
    def test(self):
        alg = ComposedAlgorithm()

        alg.eval()
        self.assertTrue(alg.alg_a._eval, "Algorithm A was not set to evaluation.")
        self.assertTrue(alg.alg_b._eval, "Algorithm B was not set to evaluation.")
        self.assertTrue(alg._eval, "Composed Algorithm was not set to evaluation.")

        alg.train()
        self.assertFalse(alg.alg_a._eval, "Algorithm A was not set to training.")
        self.assertFalse(alg.alg_b._eval, "Algorithm B was not set to training.")
        self.assertFalse(alg._eval, "Composed Algorithm was not set to training.")

class RenameTest(TestCase):
    def test(self):
        alg = TestAlgorithmA()
        alg.eval()

        metrics = alg()
        for key in metrics:
            self.assertTrue(
                alg._eval_prefix in key,
                "metrics not renamed"
            )


class TestAlgorithmA(Algorithm):
    def run(self):
        return {
            "Metric 1": torch.Tensor([random.uniform(0, 1)]),
            "Metric 2": torch.Tensor([random.uniform(0, 1)]),
            "Metric 3": torch.Tensor([random.uniform(0, 1)])
        }


class TestAlgorithmB(Algorithm):
    def run(self):
        return {
            "Metric 4": torch.Tensor([random.uniform(0, 1)]),
            "Metric 5": torch.Tensor([random.uniform(0, 1)])
        }


class TestAlgorithmC(Algorithm):
    def run(self):
        return {"Metric 6": torch.Tensor([random.uniform(0, 1)])}


class TestAlgorithmD(Algorithm):
    def run(self):
        return {"Metric 7": torch.Tensor([random.uniform(0, 1)])}


class ComposedAlgorithm(Algorithm):
    def __init__(self):
        self.alg_a = TestAlgorithmA()
        self.alg_b = TestAlgorithmB()

    def run(self):
        return self.alg_a(), self.alg_b()

def prehook(lst):
    def hook():
        lst.append('Pre Call Hook')

    return hook


def posthook(lst):
    def hook():
        lst.append('Post Call Hook')

    return hook

