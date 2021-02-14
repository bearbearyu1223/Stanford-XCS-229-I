#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
import numpy as np
import matplotlib.image as mpimg
import os
from graderUtil import graded, CourseTestRunner, GradedTestCase

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########
class Test_1a(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    self.image_small = np.copy(mpimg.imread('peppers-small.tiff'))

  @graded(is_hidden=True)
  def test_0(self):
    """1a-0-hidden: k-means (`init_centroids` shape check) """
    solution_init_centroids = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.init_centroids)

    num_clusters = 10
    student_centroids_init = submission.init_centroids(num_clusters, self.image_small.copy())
    solution_centroids_init = solution_init_centroids(num_clusters, self.image_small.copy())
    self.assertIsInstance(student_centroids_init, type(solution_centroids_init))
    self.assertTrue(student_centroids_init.shape == solution_centroids_init.shape)

  @graded(is_hidden=True)
  def test_1(self):
    """1a-1-hidden: k-means (`update_centroids` shape check)"""
    solution_init_centroids = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.init_centroids)
    solution_update_centroids = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.update_centroids)

    num_clusters = 10
    max_iter = 1
    centroids_init = solution_init_centroids(num_clusters, self.image_small)

    print("Testing Student update_centroids...")
    student_updated_centroids = submission.update_centroids(centroids_init.copy(), self.image_small.copy(),
                                                         max_iter=max_iter)
    print()
    print("Comparing with solution update_centroids...")
    solution_updated_centroids = solution_update_centroids(centroids_init.copy(), self.image_small.copy(),
                                                           max_iter=max_iter)
    self.assertIsInstance(solution_updated_centroids, type(student_updated_centroids))
    self.assertTrue(solution_updated_centroids.shape == student_updated_centroids.shape)

  @graded(is_hidden=True)
  def test_2(self):
    """1a-2-hidden:  k-means (`update_image` check)"""
    solution_init_centroids = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.init_centroids)
    solution_update_image = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.update_image)

    num_clusters = 10
    centroids_init = solution_init_centroids(num_clusters, self.image_small)

    student_updated_image = submission.update_image(self.image_small.copy(), centroids_init.copy())
    solution_updated_image = solution_update_image(self.image_small.copy(), centroids_init.copy())
    self.assertIsInstance(student_updated_image, type(solution_updated_image))
    self.assertTrue(student_updated_image.shape == solution_updated_image.shape)
    self.assertTrue(np.allclose(student_updated_image, solution_updated_image, atol=0, rtol=0.10))

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