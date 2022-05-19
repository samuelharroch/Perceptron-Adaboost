import numpy as np


class Perceptron:

    def __init__(self) -> None:
        self.dataset = np.zeros((150,3))

    def read_from_file (self, path):
        f = open(path, "r")
        for index , line in enumerate(f) :
            x,y,label = line.split(" ")
            self.dataset[index] = [x,y,label]
        f.close()
        return self.dataset

    def train(self):
        self.x_train = self.dataset[:,:2]
        self.y_train = self.dataset[:,2]

        #   perceptron :
        converge = False
        error_counter = 0
        num_of_iteration = 0
        self.W = np.zeros(2)

        while not converge:
            num_of_iteration += 1
            # assume converge
            converge = True
            #
            for point, label in zip(self.x_train, self.y_train) :

                if point.dot(self.W) > 0: predict = 1
                else: predict = -1

                if label != predict:
                    converge = False
                    error_counter += 1
                    self.W += label * point
                    break

        return self.W, error_counter, num_of_iteration

    def test(self):

        ans = self.x_train.dot(self.W.T)
        predict = np.where(ans > 0 ,1 ,-1)
        return np.equal(self.y_train,predict)


if __name__ == '__main__':
    p = Perceptron()
    p.read_from_file("two_circle.txt")
    print(p.train())
    # p.train()
    # print(p.test())