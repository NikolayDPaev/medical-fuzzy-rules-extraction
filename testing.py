def test_prediction(prediction_fun, testing_set, testing_labels):
    tp, fn, fp, tn = 0, 0, 0, 0

    for data, label in zip(testing_set, testing_labels):
        prediction = prediction_fun(data)
        if prediction == 1:
            if label == 1:
                tp += 1
            else:
                fp += 1
        else:
            if label == 1:
                fn += 1
            else:
                tn += 1
    return tp, fn, fp, tn

def print_test_results(tp, fn, fp, tn):
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    Fscore = (2.0 * precision * recall) / (precision + recall)
    print('Precision: '+str(precision))
    print('Recall: '+str(recall))
    print('F1-score: '+str(Fscore))
    print('Confusion Matrix:')
    print('{:15} {:>8} {:>8}'.format('', 'Predicted p', 'Predicted n'))
    print('{:15} {:>8.3f} {:>8.3f}'.format('Actual p', tp, fn))
    print('{:15} {:>8.3f} {:>8.3f}'.format('Actual n', fp, tn))
    return Fscore

def false_positives(prediction_fun, testing_set, testing_labels):
    false_positives = []
    for data, label in zip(testing_set, testing_labels):
        prediction = prediction_fun(data)
        if prediction == 1 and label == 0:
            false_positives.append(data)
    return false_positives

def false_negatives(prediction_fun, testing_set, testing_labels):
    false_negatives = []
    for data, label in zip(testing_set, testing_labels):
        prediction = prediction_fun(data)
        if prediction == 0 and label == 1:
            false_negatives.append(data)
    return false_negatives

def r_squared_test(prediction_fun, testing_set, testing_labels):
    sum_of_squared_regression = 0
    sum_of_squares = 0
    mean = sum(testing_labels)/len(testing_labels)

    for data, label in zip(testing_set, testing_labels):
        prediction = prediction_fun(data)
        sum_of_squared_regression += (label - prediction)**2
        sum_of_squares += (label - mean)**2

    return 1 - sum_of_squared_regression/sum_of_squares
