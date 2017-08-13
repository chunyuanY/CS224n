#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3(d): Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np
from model.GRUCell import GRUCell

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def test_gru_cell():
    with tf.Graph().as_default():
        with tf.variable_scope("test_gru_cell"):
            x_placeholder = tf.placeholder(tf.float32, shape=(None,3))
            h_placeholder = tf.placeholder(tf.float32, shape=(None,2))

            with tf.variable_scope("gru"):
                tf.get_variable("U_r", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("W_r", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_r",  initializer=np.array(np.ones(2), dtype=np.float32))
                tf.get_variable("U_z", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("W_z", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_z",  initializer=np.array(np.ones(2), dtype=np.float32))
                tf.get_variable("U_o", initializer=np.array(np.eye(3,2), dtype=np.float32))
                tf.get_variable("W_o", initializer=np.array(np.eye(2,2), dtype=np.float32))
                tf.get_variable("b_o",  initializer=np.array(np.ones(2), dtype=np.float32))

            tf.get_variable_scope().reuse_variables()
            cell = GRUCell(3, 2)
            y_var, ht_var = cell(x_placeholder, h_placeholder, scope="gru")

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                x = np.array([
                    [0.4, 0.5, 0.6],
                    [0.3, -0.2, -0.1]], dtype=np.float32)
                h = np.array([
                    [0.2, 0.5],
                    [-0.3, -0.3]], dtype=np.float32)
                y = np.array([
                    [ 0.320, 0.555],
                    [-0.006, 0.020]], dtype=np.float32)
                ht = y

                y_, ht_ = session.run([y_var, ht_var], feed_dict={x_placeholder: x, h_placeholder: h})
                print("y_ = " + str(y_))
                print("ht_ = " + str(ht_))

                assert np.allclose(y_, ht_), "output and state should be equal."
                assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."

def do_test(_):
    logger.info("Testing gru_cell")
    test_gru_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the GRU cell implemented as part of Q3 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
