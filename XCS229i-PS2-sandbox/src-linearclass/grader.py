#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
import numpy as np
from graderUtil import graded, CourseTestRunner, GradedTestCase

# Import student submission
import submission
import util

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########
class Test_1b(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    # intercept is True for logistic regression
    self.x_train_ds1, self.y_train_ds1 = util.load_dataset('ds1_train.csv', add_intercept=True)
    self.x_test_ds1, self.y_test_ds1 = util.load_dataset('ds1_test.csv', add_intercept=True)

    self.x_train_ds2, self.y_train_ds2 = util.load_dataset('ds2_train.csv', add_intercept=True)
    self.x_test_ds2, self.y_test_ds2 = util.load_dataset('ds2_test.csv', add_intercept=True)

  @graded(is_hidden=True)
  def test_0(self):
    """1b-0-hidden: logistic regression (self.theta shape check)"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LogisticRegression)
    student_clf = submission.LogisticRegression()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    solution_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    # check if student model theta is not None
    self.assertIsNotNone(student_clf.theta)
    # check if the shape is the same
    shape_matches = student_clf.theta.reshape(1, -1).shape == solution_clf.theta.reshape(1, -1).shape
    self.assertTrue(shape_matches)

  @graded(is_hidden=True)
  def test_1(self):
    """1b-1-hidden: logistic regression (self.theta check)"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LogisticRegression)
    student_clf = submission.LogisticRegression()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    solution_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    # check if student model theta is not None
    self.assertIsNotNone(student_clf.theta)
    # check if the shape is the same
    is_close = np.allclose(student_clf.theta.reshape(1, -1), solution_clf.theta.reshape(1, -1), rtol=0.25,
                           atol=0)
    self.assertTrue(is_close)

  @graded(is_hidden=True)
  def test_2(self):
    """1b-2-hidden: logistic regression (accuracy check, dataset 1 [>70%])"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LogisticRegression)
    student_clf = submission.LogisticRegression()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    solution_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    p_test = student_clf.predict(self.x_test_ds1)
    yhat = p_test > 0.5
    accuracy = np.mean((yhat == 1) == (self.y_test_ds1 == 1))
    print('logistic regression Accuracy: {}'.format(accuracy * 100))
    self.assertTrue(accuracy * 100 >= 70)

  @graded(is_hidden=True)
  def test_3(self):
    """1b-3-hidden: logistic regression (accuracy check, dataset 2 [>70%])"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LogisticRegression)
    student_clf = submission.LogisticRegression()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train_ds2.copy(), self.y_train_ds2.copy())
    solution_clf.fit(self.x_train_ds2.copy(), self.y_train_ds2.copy())
    p_test = student_clf.predict(self.x_test_ds2)
    yhat = p_test > 0.5
    accuracy = np.mean((yhat == 1) == (self.y_test_ds2 == 1))
    print('logistic regression Accuracy: {}'.format(accuracy * 100))
    self.assertTrue(accuracy * 100 >= 70)

  @graded()
  def test_4(self):
    """1b-4-basic:  Use LogReg to create a plot of dataset 1 validation set."""
    submission.main_LogReg(train_path='ds1_train.csv',
      valid_path='ds1_valid.csv',
      save_path='logreg_pred_1.txt')
    self.assertTrue(True)

class Test_1e(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    self.x_train_ds1, self.y_train_ds1 = util.load_dataset('ds1_train.csv', add_intercept=False)
    self.x_test_ds1, self.y_test_ds1 = util.load_dataset('ds1_test.csv', add_intercept=True)

    self.x_train_ds2, self.y_train_ds2 = util.load_dataset('ds2_train.csv', add_intercept=False)
    self.x_test_ds2, self.y_test_ds2 = util.load_dataset('ds2_test.csv', add_intercept=True)

  @graded(is_hidden=True)
  def test_0(self):
    """1e-0-hidden: GDA (self.theta shape check)"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.GDA)
    student_clf = submission.GDA()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    solution_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    # check if student model theta is not None
    self.assertIsNotNone(student_clf.theta)
    # check if the shape is the same
    shape_matches = student_clf.theta.reshape(1, -1).shape == solution_clf.theta.reshape(1, -1).shape
    self.assertTrue(shape_matches)

  @graded(is_hidden=True)
  def test_1(self):
    """1e-1-hidden: GDA (self.theta check)"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.GDA)
    student_clf = submission.GDA()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    solution_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    # check if student model theta is not None
    self.assertIsNotNone(student_clf.theta)
    # check if the shape is the same
    is_close = np.allclose(student_clf.theta.reshape(1, -1), solution_clf.theta.reshape(1, -1), rtol=0.25,
                           atol=0)
    self.assertTrue(is_close)

  @graded(is_hidden=True)
  def test_2(self):
    """1e-2-hidden: GDA (accuracy check, dataset 1 [>70%])"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.GDA)
    student_clf = submission.GDA()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    solution_clf.fit(self.x_train_ds1.copy(), self.y_train_ds1.copy())
    p_test = student_clf.predict(self.x_test_ds1)
    yhat = p_test > 0.5
    accuracy = np.mean((yhat == 1) == (self.y_test_ds1 == 1))
    print('GDA Accuracy: {}'.format(accuracy * 100))
    self.assertTrue(accuracy * 100 >= 70)

  @graded(is_hidden=True)
  def test_3(self):
    """1e-3-hidden: GDA (accuracy check, dataset 2 [>70%])"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.GDA)
    student_clf = submission.GDA()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train_ds2.copy(), self.y_train_ds2.copy())
    solution_clf.fit(self.x_train_ds2.copy(), self.y_train_ds2.copy())
    p_test = student_clf.predict(self.x_test_ds2)
    yhat = p_test > 0.5
    accuracy = np.mean((yhat == 1) == (self.y_test_ds2 == 1))
    print('GDA Accuracy: {}'.format(accuracy * 100))
    self.assertTrue(accuracy * 100 >= 70)

  @graded()
  def test_4(self):
    """1e-4-basic:  Use GDA to create a plot of dataset 1 validation set."""
    submission.main_GDA(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')
    self.assertTrue(True)

class Test_1g(GradedTestCase):
  @graded()
  def test_0(self):
    """1g-0-basic:  Use GDA and logreg to create a plots of datasets 1 and 2 validation sets."""
    submission.main_LogReg(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
    submission.main_GDA(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
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