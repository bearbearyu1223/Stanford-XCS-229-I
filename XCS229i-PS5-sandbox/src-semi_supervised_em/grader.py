#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
import numpy as np
from graderUtil import graded, CourseTestRunner, GradedTestCase
from sklearn.metrics import accuracy_score
import os

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################
def create_dataset(load_gmm_dataset, UNLABELED, K):
    # Load dataset
    x_all, z_all = load_gmm_dataset('train.csv')

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]  # Labeled examples
    z_tilde = z_all[labeled_idxs, :]  # Corresponding labels
    x = x_all[~labeled_idxs, :]  # Unlabeled examples

    n, d = x.shape
    group = np.random.choice(K, n)
    mu = [np.mean(x[group == g, :], axis=0) for g in range(K)]
    sigma = [np.cov(x[group == g, :].T) for g in range(K)]

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full((K,), fill_value=(1. / K), dtype=np.float32)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full((n, K), fill_value=(1. / K), dtype=np.float32)

    return {
        'x': x,
        'w': w,
        'phi': phi,
        'mu': mu,
        'sigma': sigma,
        'x_tilde': x_tilde,
        'z_tilde': z_tilde
    }

#########
# TESTS #
#########
class Test_2d(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

  @graded(is_hidden=True)
  def test_0(self):
    """2d-0-hidden:  GMM (`run_em` check)"""
    solution_run_em = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.run_em)
    load_gmm_dataset = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.load_gmm_dataset)
    UNLABELED = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.UNLABELED)
    K = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.K)

    DATASET = create_dataset(load_gmm_dataset, UNLABELED, K)

    max_iter = 10
    x = DATASET['x']
    w = DATASET['w']
    phi = DATASET['phi']
    mu = DATASET['mu']
    sigma = DATASET['sigma']
    w_student = submission.run_em(x.copy(), w.copy(), phi.copy(),
                               mu.copy(), sigma.copy(), max_iter)
    w_solution = solution_run_em(x.copy(), w.copy(), phi.copy(),
                                 mu.copy(), sigma.copy(), max_iter)

    self.assertIsInstance(w_student, type(w_solution))
    self.assertTrue(w_student.shape == w_solution.shape)
    
    student_pred = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
      student_pred[i] = np.argmax(w_student[i])

    solution_pred = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
      solution_pred[i] = np.argmax(w_solution[i])

    solution_match = accuracy_score(student_pred, solution_pred)
    self.assertTrue(round(solution_match * 100, 2) > 70)

  @graded(timeout=600)
  def test_1(self):
    """2d-1-basic:  Create plots to verify the EM implementation"""
    skip = True
    if not skip:
      np.random.seed(229)
      # Run 3 trials to see how different initializations
      # affect the final predictions with and without supervision
      for t in range(3):
        submission.main(is_semi_supervised=False, trial_num=t)
    self.assertTrue(True)

class Test_2e(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

  @graded(is_hidden=True)
  def test_0(self):
    """2e-0-hidden:  GMM (`run_semi_supervised_em` check)"""
    solution_run_semi_supervised_em = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.run_semi_supervised_em)
    load_gmm_dataset = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.load_gmm_dataset)
    UNLABELED = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.UNLABELED)
    K = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.K)

    DATASET = create_dataset(load_gmm_dataset, UNLABELED, K)

    max_iter = 10
    x = DATASET['x']
    w = DATASET['w']
    phi = DATASET['phi']
    mu = DATASET['mu']
    sigma = DATASET['sigma']
    x_tilde = DATASET['x_tilde']
    z_tilde = DATASET['z_tilde']
    w_student = submission.run_semi_supervised_em(x.copy(), x_tilde.copy(), z_tilde.copy(), w.copy(), phi.copy(),
                                               mu.copy(), sigma.copy(), max_iter)
    w_solution = solution_run_semi_supervised_em(x.copy(), x_tilde.copy(), z_tilde.copy(), w.copy(), phi.copy(),
                                                 mu.copy(), sigma.copy(), max_iter)
    self.assertIsInstance(w_student, type(w_solution))
    self.assertTrue(w_student.shape == w_solution.shape)
    student_pred = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        student_pred[i] = np.argmax(w_student[i])

    solution_pred = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        solution_pred[i] = np.argmax(w_solution[i])

    solution_match = accuracy_score(student_pred, solution_pred)
    self.assertTrue(round(solution_match * 100, 2) > 70)
  
  @graded(timeout=600)
  def test_1(self):
    """2e-1-basic:  Create plots to verify the semi-supervised EM implementation"""
    skip = True
    if not skip:
      np.random.seed(229)
      # Run 3 trials to see how different initializations
      # affect the final predictions with and without supervision
      for t in range(3):
        submission.main(is_semi_supervised=True, trial_num=t)
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