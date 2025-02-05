import csv
import numpy as np

LearnRate = 0.01

def read_file(filename):
    my_list = []
    with open(filename, mode='r') as file:
        csv_file = csv.reader(file)
        for lines in csv_file:
            my_list.append(lines)
    return my_list

def ModifyTheta(theta,right_answer,output):
    return theta - (right_answer - output) * LearnRate
def ModifyWeights(weights, right_answer, output, attributes):
    for i in range(weights.__len__()):
        weights[i] += ((right_answer - output) * LearnRate * float(attributes[i]))
    return weights

def GetOutput(weights, attributes, theta):
    scalar_product = 0
    for i in range(weights.__len__()):
        scalar_product += weights[i] * float(attributes[i])
    return 1 if scalar_product >= theta else 0


def train_perceptron(weights, points, theta, answers):
    error_sum = 0
    for point in points:
        right_answer = point[point.__len__() - 1]
        output = GetOutput(weights, point[:-1], theta)
        if output != answers.get(right_answer):
            theta = ModifyTheta(theta, answers.get(right_answer), output)
            weights = ModifyWeights(weights, answers.get(right_answer), output, point[:-1])
        error_sum += (answers.get(right_answer) - output)**2
    print(f"iter error: {int(error_sum / points.__len__() * 100)}%")
    return weights, theta
def test(weights, test_data, theta, answers):
    count = 0
    for point in test_data:
        right_answer = answers.get(point[-1])
        answer = GetOutput(weights, point[:-1], theta)
        if right_answer == answer:
            count += 1
    print(f"Accuracy: {count / test_data.__len__()}")

def main():
    filename = "perceptron"
    points = read_file(filename + ".data")
    tests = read_file(filename + ".test.data")
    weights = np.random.rand(len(points[0]) - 1)
    epochs = 100
    answers = {}
    i = 0
    for point in points:
        right_answer = point[point.__len__() - 1]
        index = answers.setdefault(right_answer, i)
        if index == i:
            i += 1
    theta = np.random.random()
    weights = []
    for _ in range(points[0].__len__() - 1):
        weights.append(np.random.random())
        for i in range(epochs):
            train_perceptron(weights, points, theta, answers)
    test(weights, tests, theta, answers)

main()