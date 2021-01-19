import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Note for enterprising students:  There are myriad ways to split sentences for
    this algorithm.  For instance, you might want to exclude punctuation (unless
    it's organized in an email address format) or exclude numbers (unless they're
    organized in a zip code or phone number format).  Clearly this can become quite
    complex.  For our purposes, please split using the space character ONLY.  This
    is intended to balance your understanding with our ability to autograde the
    assignment.  Thanks and have fun with the rest of the assignment!

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    assert isinstance(message, str), 'Type Error: input message should be string!'
    return [word.lower() for word in message.split()]
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    assert isinstance(messages, list) and all(isinstance(x,str) for x in messages), 'Type Error: input messages should be a list of string!'
    word_dictionary = {}
    corpus = []
    for msg in messages:
        words = get_words(message=msg)
        corpus = corpus + list(set(words))
    corpus_counter = collections.Counter(corpus)
    index = 0
    for word, count in corpus_counter.items():
        if count >= 5:
            word_dictionary[word] = index
            index = index + 1
    return word_dictionary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    assert isinstance(messages, list) and all(isinstance(x, str) for x in messages), 'Type Error: input messages should be a list of strings!'
    assert isinstance(word_dictionary, dict), 'Type Error: input word_dictionary should be a dict!'
    rtn = np.zeros(shape=(len(messages), len(word_dictionary)))

    for row, msg in enumerate(messages):
        all_words = get_words(message=msg)
        unique_words = set(all_words)
        for word in unique_words:
            if word in word_dictionary:
                col = word_dictionary[word]
                rtn[row][col] = all_words.count(word)
    return rtn
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    sample_size = matrix.shape[0]
    vocabulary_size = matrix.shape[1]

    is_spam_mask = labels == 1
    not_spam_mask = labels == 0
    is_spam_matrix = matrix[is_spam_mask,:]
    not_spam_matrix = matrix[not_spam_mask, :]

    phi_j_yeq1 = (np.sum(is_spam_matrix, axis=0) + 1) / (np.sum(is_spam_matrix) + vocabulary_size)
    phi_j_yeq0 = (np.sum(not_spam_matrix, axis=0) + 1) / (np.sum(not_spam_matrix) + vocabulary_size)

    phi_yeq1 = np.sum(labels==1) / sample_size
    phi_yeq0 = np.sum(labels==0) / sample_size

    model = dict()
    model['phi_yeq0'] = phi_yeq0
    model['phi_yeq1'] = phi_yeq1
    model['phi_j_yeq0'] = phi_j_yeq0
    model['phi_j_yeq1'] = phi_j_yeq1
    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    np.seterr(divide='ignore')
    p_not_spam = np.exp(np.log(model['phi_yeq0'] * np.prod(np.power(model['phi_j_yeq0'], matrix), axis=1)))
    p_is_spam = np.exp(np.log(model['phi_yeq1'] * np.prod(np.power(model['phi_j_yeq1'], matrix), axis=1)))

    prediction = []
    for p1, p2 in zip(p_is_spam, p_not_spam):
        if p1 >= p2:
            prediction.append(1)
        else:
            prediction.append(0)
    return np.array(prediction)
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    indicative_metric = np.log(np.divide(model['phi_j_yeq1'], model['phi_j_yeq0']))
    ind = indicative_metric.argsort()[-5:][::-1]
    rtn = []
    for i in np.nditer(ind):
        for k, v in dictionary.items():
            if v == i:
                rtn.append(k)
                break
    return rtn
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best_radius = 0.0
    best_accuracy = 0.0
    for r in radius_to_consider:
        pred_labels = svm.train_and_predict_svm(train_matrix=train_matrix, train_labels=train_labels, test_matrix=val_matrix, radius=r)
        accuracy = np.sum(pred_labels == val_labels)/len(val_labels)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_radius = r
    return best_radius
    # *** END CODE HERE ***

def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary_(soln)', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix_(soln)', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions_(soln)', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words_(soln)', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius_(soln)', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
