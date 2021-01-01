#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np

# Import student submission
import submission
import util

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########

class Test_2d(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    self.x_train, self.y_train = util.load_dataset('train.csv', add_intercept=False)
    self.x_eval, self.y_eval = util.load_dataset('test.csv', add_intercept=False)

  @graded(is_hidden=True)
  def test_0(self):
    """2d-0-hidden:  Poisson Regression (self.theta shape check)"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.PoissonRegression)
    student_clf = submission.PoissonRegression()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train.copy(), self.y_train.copy())
    solution_clf.fit(self.x_train.copy(), self.y_train.copy())
    # check if student model theta is not None
    self.assertIsNotNone(student_clf.theta)
    # check if the shape is the same
    shape_matches = student_clf.theta.reshape(1, -1).shape == solution_clf.theta.reshape(1, -1).shape
    self.assertTrue(shape_matches)

  @graded(is_hidden=True)
  def test_1(self):
    """2d-1-hidden:  Poisson Regression (self.theta check)"""
    solution_logreg = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.PoissonRegression)
    student_clf = submission.PoissonRegression()
    solution_clf = solution_logreg()

    student_clf.fit(self.x_train.copy(), self.y_train.copy())
    solution_clf.fit(self.x_train.copy(), self.y_train.copy())
    # check if student model theta is not None
    self.assertIsNotNone(student_clf.theta)
    # check if the shape is the same
    is_close = np.allclose(student_clf.theta.reshape(1, -1), solution_clf.theta.reshape(1, -1), rtol=0.25,
                           atol=0)
    self.assertTrue(is_close)

  @graded()
  def test_2(self):
    """2d-2-basic:  Create a plot to validate your model"""
    submission.main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
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