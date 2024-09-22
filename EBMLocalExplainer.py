from typing import Union
import os

import pandas as pd
from tqdm import tqdm
import re


class EBMLocalExplainer():
    """
    This class is used to explain the local behavior of an EBM model.
    It can be used to filter the local explanations by a word or a number (or both).

    Parameters
    ----------
    ebm_model: EBM model
        The EBM model to be explained.
    X: pandas.df
        The data used to explain the EBM model.
    y: pandas.df
        The target used to explain the EBM model.

    Methods
    -------
    add_word_filter(word: str, exact_match: bool = False)
        Add a word filter to the local explanation.
        If exact_match is True, the word will be searched as an exact match.
        If exact_match is False, the word will be searched as a substring.
    add_number_filter(number: Union[int, float])
        Add a number filter to the local explanation.
        The number will be used to filter the local explanation by the feature strength.
    add_number_word_filter(number: Union[int, float], word: str, exact_match: bool = False)
        Add a number and a word filter to the local explanation.
        The number will be used to filter the local explanation by the feature strength.
        If exact_match is True, the word will be searched as an exact match.
        If exact_match is False, the word will be searched as a substring.
    explain()
        Explain the local behavior of the EBM model, while considering the defined filters.

    More filter can be added by writing a new method with the following structure:
        1. Add the filter method (private) to the class.
            def _new_filter(self, *args):
                # *args is the list of arguments that will be passed to the method
                # The method should return True if the filter is satisfied, False otherwise.
        2. Add the filter to the list of filters by using the add_filter method.
            def add_new_filter(self, *args):
                self.filters.append((self._new_filter, *args))

        Using this kind of structure allows to add any kind of filter to the local explanation.
        (There is of course optimization potential)
    """
    def __init__(self, ebm_model, X, y):
        self.ebm_model = ebm_model
        self.X = X
        self.y = y
        self.filters = []
        self.local_explanation_data = None

        self._get_local_explanation()

    def add_word_filter(self, word: str, exact_match: bool = False) -> None:
        self.filters.append((self._word_filter, word, exact_match))

    def add_number_filter(self, number: Union[int, float]):
        self.filters.append((self._number_filter, number))

    def add_number_word_filter(self, number: Union[int, float], word: str, exact_match: bool = False):
        self.filters.append((self._number_word_filter, number, word, exact_match))

    def _get_local_explanation(self):
        self.ebm_local = self.ebm_model.explain_local(self.X, self.y, name='EBM_local')

    def _word_filter(self, *args):
        word, exact_match = args
        if exact_match:
            result = re.search((re.escape(word) + r"\s\([^)]*\)"), str(self.local_explanation_data['y']))
            if result:
                return True
        else:
            result = re.search((re.escape(word)), str(self.local_explanation_data['y']))
            if result:
                return True

    def _number_filter(self, number):
        for index, feature_strength in enumerate(self.local_explanation_data['x']):
            if abs(feature_strength) >= number and self.local_explanation_data['y'][index] != "Intercept":
                return True

    def _number_word_filter(self, *args):
        number, word, exact_match = args
        search_pattern = re.escape(word)
        if exact_match:
            search_pattern += r"\s\([^)]*\)"
        for index, feature_strength in enumerate(self.local_explanation_data['x']):
            if abs(feature_strength) >= number and re.search(search_pattern, str(self.local_explanation_data['y'][index])):
                return True

    def _get_local_explanation_dataframe(self):
        local_explaination_x = []
        for index in tqdm(self.ebm_local.selector.index):
            local_explaination_x.append([self.ebm_local.data(index)])
            local_explaination = self.ebm_local.data(index)
        local_dataframe = pd.DataFrame(columns=self.ebm_local.data(0).names)
        return self.ebm_local.data(0)

    def explain(self, verbose=False):
        #self._get_local_explanation_dataframe() # work-in-progress

        # check if path plots/local exists
        if not os.path.exists("plots/local"):
            os.makedirs("plots/local")
        print("\n\n Looking for local explanations considering the defined filters.\n")
        for index in tqdm(self.ebm_local.selector.index):
            if len(self.filters) == 0:
                plotly_fig = self.ebm_local.visualize(index)
                plotly_fig.write_image(f"plots/local/EBM_local_{index}.png")
            else:
                self.local_explanation_data = {
                    'x': self.ebm_local.visualize(index).data[0]['x'],
                    'y': self.ebm_local.visualize(index).data[0]['y']
                }
                filter_true = []
                for filter, *filter_arguments in self.filters:
                    if filter(*filter_arguments):
                        filter_true.append(True)
                    else:
                        filter_true.append(False)
                if all(filter_true):
                    print("Found local explanation for filter: ", *filter_arguments)
                    plotly_fig = self.ebm_local.visualize(index)
                    plotly_fig.write_image(f"plots/local/EBM_local_{index}.png")
                else:
                    falsy_filter = []
                    for i in range(len(filter_true)):
                        if filter_true[i] == True:
                            continue
                        else:
                            falsy_filter.append(self.filters[i])
                    if verbose:
                        print(f'\n No local explanation found for filter: \n \
                        {list(zip(*falsy_filter))[2]}\n \
                        {list(zip(*falsy_filter))[1]}\n')