#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections
import numpy as np
from graderUtil import graded, CourseTestRunner, GradedTestCase

# Import student submission
import submission
import util
import svm

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########
class Test_1a(GradedTestCase):
  def setUp(self):
    self.train_messages, self.train_labels = util.load_spam_dataset('spam_train.tsv')

  @graded(is_hidden=True)
  def test_0(self):
    """1a-0-hidden: spam (`get_words` check)"""
    test_message = np.random.choice(self.train_messages)
    solution_get_words = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.get_words)

    student_result = submission.get_words(message=test_message)
    solution_result = solution_get_words(message=test_message)

    self.assertTrue(collections.Counter(student_result) == collections.Counter(solution_result))

  @graded(is_hidden=True)
  def test_1(self):
    """1a-1-hidden: spam (`create_dictionary` check)"""
    solution_create_dictionary = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.create_dictionary)

    student_result = submission.create_dictionary(messages=self.train_messages)
    solution_result = solution_create_dictionary(messages=self.train_messages)

    self.assertTrue(set(student_result.keys()) == set(solution_result.keys()))

  @graded(is_hidden=True)
  def test_2(self):
    """1a-2-hidden: spam (`transform_text` output shape check)"""
    solution_create_dictionary = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.create_dictionary)
    solution_transform_text = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.transform_text)

    dictionary = solution_create_dictionary(self.train_messages)

    student_result = submission.transform_text(messages=self.train_messages, word_dictionary=dictionary)
    solution_result = solution_transform_text(messages=self.train_messages, word_dictionary=dictionary)
    self.assertTrue(student_result.shape == solution_result.shape)

  @graded(is_hidden=True)
  def test_3(self):
    """1a-3-hidden: spam (Size of dictionary check)"""
    solution_create_dictionary = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.create_dictionary)

    student_result = submission.create_dictionary(messages=self.train_messages)
    solution_result = solution_create_dictionary(messages=self.train_messages)

    self.assertTrue(len(student_result) == len(solution_result))

  @graded()
  def test_4(self):
    """1a-4-basic:  Create the spam_dictionary and spam_sample_train_matrix files"""
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = submission.create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary_(soln)', dictionary)

    train_matrix = submission.transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix_(soln)', train_matrix[:100,:])
    self.assertTrue(True)

class Test_1b(GradedTestCase):
  def setUp(self):
    self.train_messages, self.train_labels = util.load_spam_dataset('spam_train.tsv')
    self.test_messages, self.test_labels = util.load_spam_dataset('spam_test.tsv')

  @graded(is_hidden=True)
  def test_0(self):
    """1b-0-hidden: spam (`fit_naive_bayes_model` accuracy check [>80%])"""
    solution_create_dictionary = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.create_dictionary)
    solution_transform_text = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.transform_text)

    dictionary = solution_create_dictionary(self.train_messages)
    train_matrix = solution_transform_text(self.train_messages, dictionary)
    test_matrix = solution_transform_text(self.test_messages, dictionary)

    student_model = submission.fit_naive_bayes_model(train_matrix, self.train_labels)
    student_predictions = submission.predict_from_naive_bayes_model(student_model, test_matrix)

    student_model_accuracy = np.mean(student_predictions == self.test_labels)
    print('Naive Bayes had an accuracy of {} on a sample testing set'.format(round(student_model_accuracy * 100, 2)))
    self.assertTrue(student_model_accuracy * 100 >= 80)

  @graded()
  def test_1(self):
    """1b-1-basic:  compute naive bayes prediction accuracy and then save resulting predictions"""
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = submission.create_dictionary(train_messages)
    train_matrix = submission.transform_text(train_messages, dictionary)
    val_matrix = submission.transform_text(val_messages, dictionary)
    test_matrix = submission.transform_text(test_messages, dictionary)

    naive_bayes_model = submission.fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = submission.predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions_(soln)', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    self.assertTrue(True)

class Test_1c(GradedTestCase):
  def setUp(self):
    self.train_messages, self.train_labels = util.load_spam_dataset('spam_train.tsv')
    self.test_messages, self.test_labels = util.load_spam_dataset('spam_test.tsv')

  @graded(is_hidden=True)
  def test_0(self):
    """1c-0-hidden: spam (`get_top_five_naive_bayes_words`)"""
    solution_create_dictionary = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.create_dictionary)
    solution_transform_text = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.transform_text)
    solution_fit_naive_bayes_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.fit_naive_bayes_model)
    solution_get_top_five_naive_bayes_words = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.get_top_five_naive_bayes_words)

    dictionary = solution_create_dictionary(self.train_messages)
    train_matrix = solution_transform_text(self.train_messages, dictionary)

    solution_model = solution_fit_naive_bayes_model(train_matrix, self.train_labels)
    student_model = submission.fit_naive_bayes_model(train_matrix, self.train_labels)

    top_5_words_student = submission.get_top_five_naive_bayes_words(student_model, dictionary)
    top_5_words_solution = solution_get_top_five_naive_bayes_words(solution_model, dictionary)

    matches = list(set(top_5_words_student) & set(top_5_words_solution))
    print("Top 5 words : {}".format(" , ".join(top_5_words_student)))
    print("Top words that match the solution : {}".format(" , ".join(matches)))
    
    self.assertTrue(len(matches) == 5)

  @graded()
  def test_1(self):
    """1c-1-basic:  Calculate the top five most indicative words"""
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    dictionary = submission.create_dictionary(train_messages)
    train_matrix = submission.transform_text(train_messages, dictionary)
    naive_bayes_model = submission.fit_naive_bayes_model(train_matrix, train_labels)
    top_5_words = submission.get_top_five_naive_bayes_words(naive_bayes_model, dictionary)
    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)
    util.write_json('spam_top_indicative_words_(soln)', top_5_words)
    self.assertTrue(True)

class Test_1d(GradedTestCase):
  def setUp(self):
    self.train_messages, self.train_labels = util.load_spam_dataset('spam_train.tsv')
    self.val_messages, self.val_labels = util.load_spam_dataset('spam_val.tsv')
    self.test_messages, self.test_labels = util.load_spam_dataset('spam_test.tsv')

  @graded(is_hidden=True, timeout=45)
  def test_0(self):
    """1d-0-hidden: svm (`compute_best_svm_radius`)"""
    solution_create_dictionary = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.create_dictionary)
    solution_transform_text = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.transform_text)

    dictionary = solution_create_dictionary(self.train_messages)
    train_matrix = solution_transform_text(self.train_messages, dictionary)
    val_matrix = solution_transform_text(self.val_messages, dictionary)

    optimal_radius = submission.compute_best_svm_radius(train_matrix, self.train_labels, val_matrix, self.val_labels,
                                                     [0.01, 0.1, 1, 10])

    self.assertTrue(optimal_radius == 0.1)

  @graded(is_hidden=True, timeout=60)
  def test_1(self):
    """1d-1-hidden: svm (Accuracy > 95% on a sample test set trained on SVM with `optimal_radius` computed using `compute_best_svm_radius`)"""
    solution_create_dictionary = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.create_dictionary)
    solution_transform_text = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.transform_text)

    dictionary = solution_create_dictionary(self.train_messages)
    train_matrix = solution_transform_text(self.train_messages, dictionary)
    val_matrix = solution_transform_text(self.val_messages, dictionary)
    test_matrix = solution_transform_text(self.test_messages, dictionary)

    optimal_radius = submission.compute_best_svm_radius(train_matrix, self.train_labels, val_matrix, self.val_labels,
                                                   [0.01, 0.1, 1, 10])
    svm_predictions = svm.train_and_predict_svm(train_matrix, self.train_labels, test_matrix, optimal_radius)
    svm_accuracy = np.mean(svm_predictions == self.test_labels)

    print("SVM accuracy : {}".format(round(svm_accuracy*100, 2)))

    self.assertTrue(svm_accuracy*100 > 95)

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)