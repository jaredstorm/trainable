from unittest import TestCase

from trainable.progress import ProgressManager
import random


class ProgressManagerTest(TestCase):

    def test_basic_functionality(self):
        loop = ProgressManager()
        passed = True

        loop.start(1000, 0)
        for i in range(1, 1001):
            updated = loop.update(i)
            passed = passed if updated else False
        loop.end()

        self.assertTrue(passed)

    def test_update_frequency(self):
        loop = ProgressManager()

        total = 10000
        freq = 10

        loop.set_frequency(freq)
        loop.start(total, 1)
        for i in range(1, total+1):
            expect_update = i % freq == 0
            pre = f'After {i} steps, '
            post1 = 'display updated at incorrect time.'
            post2 = 'display did not update on expected step.'
            self.assertTrue(
                loop.update(i) == expect_update,
                pre + (post2 if expect_update else post1)
            )

    def test_metrics_reporting(self):
        loop = ProgressManager()
        total = 10000
        freq = 10
        loop.set_frequency(freq)
        loop.start(total, 1)
        for i in range(1, total+1):
            loop.update(i, {
                'Loss': random.randint(1, 10),
                'Other Loss': random.randint(1, 10),
                'Bacon': random.randint(1, 10)
            })
        self.assertTrue(True)
