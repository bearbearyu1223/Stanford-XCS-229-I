#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections
import numpy as np
from graderUtil import graded, CourseTestRunner, GradedTestCase

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################
N_ROWS = 5000

#########
# TESTS #
#########
class Test_2ai(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

  @graded(is_hidden=True)
  def test_0(self):
    """2ai-0-hidden: nn (`softmax`)"""
    solution_softmax = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.softmax)

    x = np.random.normal(size=(100, 10))
    student_result = submission.softmax(x)
    solution_result = solution_softmax(x)

    self.assertTrue(np.allclose(student_result, solution_result, atol=0, rtol=0.05))

  @graded(is_hidden=True)
  def test_1(self):
    """2ai-1-hidden: nn (`sigmoid`)"""
    solution_sigmoid = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.sigmoid)

    x = np.random.normal(size=(100, 10))

    student_result = submission.sigmoid(x)
    solution_result = solution_sigmoid(x)

    self.assertTrue(np.allclose(student_result, solution_result, atol=0, rtol=0.05))

  @graded(is_hidden=True)
  def test_2(self):
    """2ai-2-hidden: nn (`get_initial_params` output shape check)"""
    solution_get_initial_params = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.get_initial_params)

    input_size = 100
    num_hidden = 10
    num_output = 2

    student_result = submission.get_initial_params(input_size=input_size, num_hidden=num_hidden, num_output=num_output)
    solution_result = solution_get_initial_params(input_size=input_size, num_hidden=num_hidden,
                                                  num_output=num_output)
    results_match = True
    if student_result['W1'].shape != solution_result['W1'].shape:
      print('Got {} shape for W1 which is incorrect'.format(solution_result['W1'].shape))
      results_match = False

    if student_result['b1'].shape != solution_result['b1'].shape:
      print('Got {} shape for b1 which is incorrect'.format(solution_result['b1'].shape))
      results_match = False

    if student_result['W2'].shape != solution_result['W2'].shape:
      print('Got {} shape for W2 which is incorrect'.format(solution_result['W2'].shape))
      results_match = False

    if student_result['b2'].shape != solution_result['b2'].shape:
      print('Got {} shape for b2 which is incorrect'.format(solution_result['b2'].shape))
      results_match = False

    self.assertTrue(results_match)

class Test_2aii(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    self.train_data, self.train_labels = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.read_data('images_train.csv', 'labels_train.csv', max_rows=N_ROWS))
    self.train_labels = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.one_hot_labels(self.train_labels))

  @graded(is_hidden=True)
  def test_3(self):
    """2aii-3-hidden: nn (`forward_prop`)"""
    solution_get_initial_params = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.get_initial_params)
    solution_forward_prop = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.forward_prop)

    (nexp, dim) = self.train_data.shape
    num_hidden = 300

    params = solution_get_initial_params(input_size=dim, num_hidden=num_hidden, num_output=10)

    h_student, y_student, cost_student = submission.forward_prop(data=self.train_data[:1000, :],
                                                              labels=self.train_labels[:1000, :], params=params)
    h_solution, y_solution, cost_solution = solution_forward_prop(data=self.train_data[:1000, :],
                                                                  labels=self.train_labels[:1000, :],
                                                                  params=params)

    result_match = all([np.allclose(h_student, h_solution, atol=0, rtol=0.10),
                        np.allclose(y_student, y_solution, atol=0, rtol=0.10),
                        np.allclose(cost_solution, cost_student, atol=0, rtol=0.10)])

    self.assertTrue(result_match)

  @graded(is_hidden=True, timeout=10)
  def test_4(self):
    """2aii-4-hidden: nn (`backward_prop`)"""
    solution_get_initial_params = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.get_initial_params)
    solution_forward_prop = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.forward_prop)
    solution_backward_prop = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.backward_prop)

    (nexp, dim) = self.train_data.shape
    num_hidden = 300

    params = solution_get_initial_params(input_size=dim, num_hidden=num_hidden, num_output=10)
    student_grad = submission.backward_prop(data=self.train_data, labels=self.train_labels, params=params,
                                         forward_prop_func=solution_forward_prop)
    solution_grad = solution_backward_prop(data=self.train_data, labels=self.train_labels, params=params,
                                             forward_prop_func=solution_forward_prop)

    results_match = True
    if not np.allclose(student_grad['W1'], solution_grad['W1'], atol=0, rtol=0.20):
      print('W1 weight did not match expected result')
      results_match = False

    if not np.allclose(student_grad['b1'], solution_grad['b1'], atol=0, rtol=0.20):
      print('b1 bias did not match expected result')
      results_match = False

    if not np.allclose(student_grad['W2'], solution_grad['W2'], atol=0, rtol=0.20):
      print('W2 weight did not match expected result')
      results_match = False

    if not np.allclose(student_grad['b2'], solution_grad['b2'], atol=0, rtol=0.20):
      print('b2 bias did not match expected result')
      results_match = False

    self.assertTrue(results_match)

  @graded(is_hidden=True, timeout=60)
  def test_5(self):
    """2aii-5-hidden: nn (`gradient_descent_epoch`)"""
    solution_get_initial_params = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.get_initial_params)
    solution_forward_prop = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.forward_prop)
    solution_backward_prop = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.backward_prop)
    solution_gradient_descent_epoch = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.gradient_descent_epoch)

    (nexp, dim) = self.train_data.shape
    num_hidden = 300

    student_params = solution_get_initial_params(input_size=dim, num_hidden=num_hidden, num_output=10)
    solution_params = copy.deepcopy(student_params)

    # updates params
    submission.gradient_descent_epoch(self.train_data, self.train_labels, learning_rate=0.01, batch_size=5,
                                   params=student_params,
                                   forward_prop_func=solution_forward_prop,
                                   backward_prop_func=solution_backward_prop)
    solution_gradient_descent_epoch(self.train_data, self.train_labels, learning_rate=0.01, batch_size=5,
                                   params=solution_params,
                                   forward_prop_func=solution_forward_prop,
                                   backward_prop_func=solution_backward_prop)

    results_match = True
    if not np.allclose(student_params['W1'], solution_params['W1'], atol=0, rtol=0.20):
      print('W1 weight did not match expected result')
      results_match = False

    if not np.allclose(student_params['b1'], solution_params['b1'], atol=0, rtol=0.20):
      print('b1 bias did not match expected result')
      results_match = False

    if not np.allclose(student_params['W2'], solution_params['W2'], atol=0, rtol=0.20):
      print('W2 weight did not match expected result')
      results_match = False

    if not np.allclose(student_params['b2'], solution_params['b2'], atol=0, rtol=0.20):
      print('b2 bias did not match expected result')
      results_match = False

    self.assertTrue(results_match)

  @graded(timeout = 300)
  def test_6(self):
    """2aii-6-basic: Train the model and plot the results (not regularized)"""
    # Change this to "skip = False" to train the model.
    # This is turned off by default to make the autograder faster.
    skip = False
    if not skip:
      submission.main(train_baseline=True, train_regularized=False)
    self.assertTrue(True)

class Test_2b(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    self.train_data, self.train_labels = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.read_data('images_train.csv', 'labels_train.csv', max_rows=N_ROWS))
    self.train_labels = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.one_hot_labels(self.train_labels))

  @graded(is_hidden=True)
  def test_0(self):
    """2b-0-hidden: nn (`backward_prop_regularized`)"""
    solution_backward_prop_regularized = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.backward_prop_regularized)
    solution_forward_prop = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.forward_prop)

    (nexp, dim) = self.train_data.shape
    num_hidden = 300

    params = submission.get_initial_params(input_size=dim, num_hidden=num_hidden, num_output=10)
    student_grad = submission.backward_prop_regularized(data=self.train_data, labels=self.train_labels, params=params,
                                                     forward_prop_func=solution_forward_prop, reg=0.0001)
    solution_grad = solution_backward_prop_regularized(data=self.train_data, labels=self.train_labels, params=params,
                                                       forward_prop_func=solution_forward_prop, reg=0.0001)
    results_match = True
    if not np.allclose(student_grad['W1'], solution_grad['W1'], atol=0, rtol=0.20):
      print('W1 weight did not match expected result')
      results_match = False

    if not np.allclose(student_grad['b1'], solution_grad['b1'], atol=0, rtol=0.20):
      print('b1 bias did not match expected result')
      results_match = False

    if not np.allclose(student_grad['W2'], solution_grad['W2'], atol=0, rtol=0.20):
      print('W2 weight did not match expected result')
      results_match = False

    if not np.allclose(student_grad['b2'], solution_grad['b2'], atol=0, rtol=0.20):
      print('b2 bias did not match expected result')
      results_match = False

    self.assertTrue(results_match)

  @graded(is_hidden=True)
  def test_1(self):
    """2b-1-hidden: nn (`gradient_descent_epoch` with regularization)"""
    solution_backward_prop_regularized = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.backward_prop_regularized)
    solution_get_initial_params = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.get_initial_params)
    solution_gradient_descent_epoch = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.gradient_descent_epoch)
    solution_forward_prop = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.forward_prop)

    (nexp, dim) = self.train_data.shape
    num_hidden = 300
    student_params = solution_get_initial_params(input_size=dim, num_hidden=num_hidden, num_output=10)
    solution_params = copy.deepcopy(student_params)

    # updates params
    submission.gradient_descent_epoch(self.train_data, self.train_labels, learning_rate=0.01, batch_size=5,
                                   params=student_params,
                                   forward_prop_func=solution_forward_prop,
                                   backward_prop_func=lambda a, b, c, d: submission.backward_prop_regularized(a, b,
                                                                                                           c,
                                                                                                           d,
                                                                                                           reg=0.0001))
    solution_gradient_descent_epoch(self.train_data, self.train_labels, learning_rate=0.01, batch_size=5,
                                    params=solution_params,
                                    forward_prop_func=solution_forward_prop,
                                    backward_prop_func=lambda a, b, c, d: solution_backward_prop_regularized(a,
                                                                                                             b,
                                                                                                             c,
                                                                                                             d,
                                                                                                             reg=0.0001))
    results_match = True
    if not np.allclose(student_params['W1'], solution_params['W1'], atol=0, rtol=0.20):
      print('W1 weight did not match expected result')
      results_match = False

    if not np.allclose(student_params['b1'], solution_params['b1'], atol=0, rtol=0.20):
      print('b1 bias did not match expected result')
      results_match = False

    if not np.allclose(student_params['W2'], solution_params['W2'], atol=0, rtol=0.20):
      print('W2 weight did not match expected result')
      results_match = False

    if not np.allclose(student_params['b2'], solution_params['b2'], atol=0, rtol=0.20):
      print('b2 bias did not match expected result')
      results_match = False

    self.assertTrue(results_match)

  @graded(timeout = 300)
  def test_2(self):
    """2b-2-basic: Train the model and plot the results (regularized)"""
    # Change this to "skip = False" to train the model.
    # This is turned off by default to make the autograder faster.
    skip = False
    if not skip:
      submission.main(train_baseline=False, train_regularized=True)
    self.assertTrue(True)

class Test_2c(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

  @graded(is_hidden=True, timeout=300)
  def test_0(self):
    """2c-0-hidden: nn (regularized and non-regularized test accuracy [>= 70%])"""
    def train_model():
      train_data, train_labels = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.read_data('images_train.csv', 'labels_train.csv', max_rows=N_ROWS))
      train_labels = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.one_hot_labels(train_labels))

      test_data, test_labels = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.read_data('images_test.csv', 'labels_test.csv'))
      test_labels = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.one_hot_labels(test_labels))

      p = np.random.permutation(train_data.shape[0])
      train_data = train_data[p, :]
      train_labels = train_labels[p, :]

      dev_data = train_data[0:100, :]
      dev_labels = train_labels[0:100, :]
      train_data = train_data[100:, :]
      train_labels = train_labels[100:, :]

      mean = np.mean(train_data)
      std = np.std(train_data)
      train_data = (train_data - mean) / std
      dev_data = (dev_data - mean) / std
      test_data = (test_data - mean) / std

      all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
      }

      all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
      }

      def run_train_test(all_data, all_labels, backward_prop, num_epochs):
        params, cost_train, cost_dev, accuracy_train, accuracy_dev = submission.nn_train(
          all_data['train'], all_labels['train'],
          all_data['dev'], all_labels['dev'],
          submission.get_initial_params, submission.forward_prop, backward_prop,
          num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
        )
        accuracy = submission.nn_test(all_data['test'], all_labels['test'], params)

        return accuracy

      baseline_acc = run_train_test(all_data, all_labels, submission.backward_prop, 30)
      reg_acc = run_train_test(all_data, all_labels,
                               lambda a, b, c, d: submission.backward_prop_regularized(a, b, c, d, reg=0.0001),
                               30)

      return baseline_acc, reg_acc

    baseline_acc, reg_acc = train_model()

    print("Baseline model accuracy: {}".format(round(baseline_acc * 100, 2)))
    print("Regularized model accuracy : {}".format(round(reg_acc * 100, 2)))
    self.assertTrue(round(baseline_acc * 100, 2) >= 70)
    self.assertTrue(round(reg_acc * 100, 2) >= 70)

  @graded(timeout = 300)
  def test_1(self):
    """2c-1-basic: Train the model and plot the results (both regularized and not regularized), comparing with test set."""
    # Change this to "skip = False" to train the model.
    # This is turned off by default to make the autograder faster.
    skip = False
    if not skip:
      submission.main(train_baseline=True, train_regularized=True, test_set = True)
    self.assertTrue(True)

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