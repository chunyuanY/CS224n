#!/usr/bin/env python

import numpy as np
import random
from random import randrange

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        old = x[ix]
        x[ix] = old - h
        random.setstate(rndstate)  # 这行代码一定要写，否则后面q3_word2vec.py的测试不能通过。
        fxsubh, _ = f(x)

        x[ix] = old + h
        random.setstate(rndstate) # 这行代码一定要写，否则后面q3_word2vec.py的测试不能通过。
        fxplush, _ = f(x)

        numgrad = (fxplush - fxsubh)/(2*h)
        x[ix] = old
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return
        it.iternext() # Step to next dimension

    print("Gradient check passed!")


def gradcheck_sparse(f, x, num_checks=10, h=1e-4):
    """
    sample a few random elements and only return numerical
    in this dimensions.
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    _, analytic_grad = f(x)
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        random.setstate(rndstate)  # 这行代码一定要写，否则后面q3_word2vec.py的测试不能通过。
        fxph, _ = f(x) # evaluate f(x + h)
        x[ix] = oldval - h # increment by h
        random.setstate(rndstate)  # 这行代码一定要写，否则后面q3_word2vec.py的测试不能通过。
        fxmh, _ = f(x) # evaluate f(x - h)
        x[ix] = oldval # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = (abs(grad_numerical - grad_analytic) /
                    (abs(grad_numerical) + abs(grad_analytic)))
        print('numerical: %f analytic: %f, relative error: %e'
              %(grad_numerical, grad_analytic, rel_error))


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))  # scalar test
    gradcheck_sparse(quad, np.array(123.456))
    gradcheck_naive(quad, np.random.randn(3, ))  # 1-D test
    gradcheck_sparse(quad, np.random.randn(3, ))
    gradcheck_naive(quad, np.random.randn(4, 5))  # 2-D test
    gradcheck_sparse(quad, np.random.randn(4, 5))
    print()


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    quad = lambda x: (np.sum(np.exp(x)+x**3+5*x), np.exp(x)+3*x**2+5)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))  # scalar test
    gradcheck_naive(quad, np.random.randn(3, ))  # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))  # 2-D test
    print()
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    print("=============")
    your_sanity_checks()
