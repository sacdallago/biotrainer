import unittest

from functools import reduce
from itertools import combinations
from biotrainer.trainers import hp_manager


class HyperParameterSearchTests(unittest.TestCase):
    def test_hps_grid(self):
        # Only normal lists
        param_dict = {
            "cross_validation_config": {
                "method": "k_fold",
                "search_method": "grid_search"
            },
            "optimizer_choice": "adam",
            "use_class_weights": [True, False],
            "batch_size": [16, 48, 128],
            "learning_rate": [1e-3, 1e-4, 1e-5]
        }
        hps = hp_manager.HyperParameterManager(**param_dict)
        number_all_combinations = reduce(lambda l1, l2: l1 * l2,
                                         map(len, [val for val in param_dict.values()
                                                   if val != "adam" and type(val) != dict]))
        self.assertTrue(len(list(hps._grid_search())) == number_all_combinations)

        # Range and list comprehension
        param_dict = {
            "cross_validation_config": {
                "method": "k_fold",
                "search_method": "grid_search"
            },
            "optimizer_choice": "adam",
            "use_class_weights": [True, False],
            "batch_size": "range(0, 10, 1)",
            "learning_rate": "[10**-x for x in [3,4,5]]"
        }
        hps = hp_manager.HyperParameterManager(**param_dict)
        number_all_combinations = reduce(lambda l1, l2: l1 * l2, map(len, [[True, False], list(range(0, 10, 1)),
                                                                           [10 ** -x for x in [3, 4, 5]]]))
        self.assertTrue(len(list(hps._grid_search())) == number_all_combinations)

    def test_hps_random(self):
        n_max_evaluations = 10
        param_dict = {
            "optimizer_choice": "adam",
            "cross_validation_config": {
                "method": "k_fold",
                "search_method": "random_search",
                "n_max_evaluations_random": n_max_evaluations
            },
            "use_class_weights": [True, False],
            "batch_size": "range(0, 10, 1)",
            "learning_rate": "[10**-x for x in [3,4,5]]"
        }
        hps = hp_manager.HyperParameterManager(**param_dict)
        hyper_params_returned = []
        for hyper_params in hps._random_search():
            hyper_params_returned.append(hyper_params)
        self.assertTrue(len(hyper_params_returned) <= n_max_evaluations,
                        "random_search did not return correct amount of combinations!")

        all_combinations = combinations(hyper_params_returned, 2)
        all_unequal = all(map(lambda dicts: dicts[0] != dicts[1], all_combinations))
        self.assertTrue(all_unequal, msg="Random search produces redundant hyper parameter combinations!")


if __name__ == '__main__':
    unittest.main()
