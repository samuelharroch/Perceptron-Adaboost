import itertools
import random
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

EPS = 0.0000000000001


def find_diff(l1: List, l2: List):
    from collections import Counter

    c1 = Counter(l1)
    c2 = Counter(l2)

    diff = c1 - c2
    return list(diff.elements())


class Point:
    def __init__(self,x, y, d):
        self.x = x
        self.y = y
        self.d = d

    def __str__(self):
        return f"x={self.x}, y={self.y}, d={self.d}"

    def __repr__(self):
        return self.__str__()


class Adaboost:

    def __init__(self) -> None:
        self.dataset = []

    def read_from_file (self, path):
        f = open(path, "r")
        for index , line in enumerate(f) :
            x,y,label = line.split()
            point = Point(float(x),float(y), 1/150.0)
            self.dataset.append([point,int(label)])
        f.close()

        self.dataset = np.array(self.dataset)
        return self.dataset

    def split_data(self):
        return train_test_split(self.dataset[:,:1], self.dataset[:,1], test_size=0.5, random_state=42)

    def define_rules(self, X_train):    # If f(x) is right to the line we define her as + otherwise -
        H = []

        for p1, p2 in itertools.combinations(X_train,2):
            x1, y1 = p1[0].x , p1[0].y
            x2, y2 = p2[0].x , p2[0].y

            if (x1,y1) == (x2,y2):
                continue

            if x1 == x2:
                f1 = lambda point: 1 if x1 >= point.x else -1
                f2 = lambda point: 1 if x1 <= point.x else -1
            else:
                gradient = (y1-y2) / (x1- x2)
                n = y1 - gradient*x1
                func = lambda x: gradient*x + n
                f1 = lambda point: 1 if func(point.x) >= point.y else -1
                f2 = lambda point: 1 if func(point.x) <= point.y else -1

            H.append(f1)
            H.append(f2)
        return H

    def run_adaboost(self, H, X_train, y_train):
        errors = []
        for h in H:
            e = sum([x[0].d*(0 if h(x[0])==y else 1 ) for x,y in zip(X_train, y_train)])
            errors.append([h,e])
        errors.sort(key=lambda t: t[1])
        best_rules8 = errors[:8]
        weights = [0.5*np.log((1- e)/(e+EPS)) for h,e in best_rules8 ]

        normalizer = sum([x[0].d*np.exp(-sum([alpha*y*he[0](x[0]) for alpha, he in zip(weights,best_rules8)])) for x,y in zip(X_train, y_train)])
        for x,y in zip(X_train, y_train):
            x[0].d = x[0].d*np.exp(-sum([alpha*y*he[0](x[0]) for alpha, he in zip(weights,best_rules8)]))/ normalizer

        return best_rules8, weights

    def train(self,k=50):
        empirical_error = np.zeros(8)
        true_error = np.zeros(8)

        for i in range(k):
            X_train, X_test, y_train, y_test = self.split_data()
            H = self.define_rules(X_train)

            best_rules, alpha = self.run_adaboost(H=H, X_train=X_train, y_train=y_train)

            for j in range(8):
                Hj = lambda x: np.sign(sum([a*h[0](x[0]) for a, h in zip(alpha[:j], best_rules[:j])]))
                empirical_error[j] += np.mean([0 if Hj(x) == y else 1 for x, y in zip(X_train, y_train) ])*(1/k)
                true_error[j] += np.mean([0 if Hj(x) == y else 1 for x, y in zip(X_test, y_test)])*(1/k)

        return empirical_error, true_error


if __name__ == '__main__':
    a = Adaboost()
    a.read_from_file('four_circle.txt')
    print(a.train())
    # l1 = list(range(150))
    # l2 = random.sample(range(150), 75)
    # l2.sort()
    # l1 = find_diff(l1, l2)
    # l1.sort()
    # print(l1)
    # print(l2)