import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class SupportVectorMachine:

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = { 1: 'r', -1: 'b' }

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.data = None
        self.max_feature_value = None
        self.min_feature_value = None

    # train
    def fit(self, data):
        self.data = data

        # { ||w||: [w, b] }
        opt_dict = {}
        transforms = [
            [1,1],
            [-1,1],
            [1,-1],
            [1,1]
        ]
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = max(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]

        # extremely expensive --> takes very long
        b_range_multiple = 5 # what we are adding to while moving b around
        b_multiple = 5

        latest_optimum = self.max_feature_value*10 # starting value for W

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum]) # SEE note.txt

            # we can know if we are optimized since it is a convex problem
            optimized = False

            while not optimized:
                pass



    def predict(self, features):
        return np.sign(np.dot(np.array(features), self.w) + self.b) # sign(x.w + b)



data_dict = {
    -1: np.array([
        [1, 7],
        [2, 8],
        [3, 8]
    ]),
    1: np.array([
        [5, 1],
        [6, -1],
        [7, 3]
    ])
}
