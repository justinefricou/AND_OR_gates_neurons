import math
import random

class Neuron:

    def __init__(self):
        self._activate = 0
        self._gradient = 0
        self._bias = random.uniform(-1, 1)
        self._weights = [random.uniform(-1, 1), random.uniform(-1, 1)]

    def activation(self, input):
        sp = self._bias
        sp += input[0] * self._weights[0] + input[1] * self._weights[1]
        self._activate =  1 / (1 + math.exp(-sp))

    def compute_gradient(self, result):
        self._gradient = result - self._activate

    def update_weights(self, input, learning_rate):
        self._bias = self._bias + (self._gradient * learning_rate)
        for i in range(len(self._weights)):
            self._weights[i] = self._weights[i]
            self._weights[i] += self._gradient * learning_rate * input[i]

    def train(self, inputs, expected_results, learning_rate):
        for _ in range(100):
            for i in range (len(inputs)):
                self.activation(inputs[i])
                self.compute_gradient(expected_results[i])
                self.update_weights(inputs[i], learning_rate)
        results = []
        for i in range(len(inputs)):
            self.activation(inputs[i])
            results.append(self._activate)
        return results

def main():
    inputs = [[0, 0], [1, 0], [0, 1], [1, 1]]
    learning_rate = 0.5
    expected_results_and = [0, 0, 0, 1]
    expected_results_or = [0, 1, 1, 1]

    neuron_and = Neuron()
    results_and = neuron_and.train(inputs, expected_results_and, learning_rate)
    print("\nAnd gate :")
    for i in range(4):
        print("input : {} -> output : {}".format(inputs[i], results_and[i]))

    neuron_or = Neuron()
    results_or = neuron_or.train(inputs, expected_results_or, learning_rate)
    print("\nOr gate :")
    for i in range(4):
        print("input : {} -> output : {}".format(inputs[i], results_or[i]))

if __name__ == "__main__":
    main()