from typing import List, Tuple

import numpy as np
import scipy


def check_mat_sizes():
    """
    X: m x n
    W: k x k
    Z = X * W

    combinations of m x n with k

    m         n         k
    even  ==  even      even
    even  ==  even      odd
    even  !=  even      even
    even  !=  even      odd
    even      odd       even
    even      odd       odd
    odd       even      even
    odd       even      odd
    odd   ==  odd       even
    odd   ==  odd       odd
    odd   !=  odd       even
    odd   !=  odd       odd

    :return:
    """

    test_types: List[str] = [
        "m (even) == n (even), k (even)",
        "m (even) == n (even), k (odd)",
        "m (even) != n (even), k (even)",
        "m (even) != n (even), k (odd)",
        "m (even)    n (odd) , k (even)",
        "m (even)    n (odd) , k (odd)",
        "m (odd)     n (even), k (even)",
        "m (odd)     n (even), k (odd)",
        "m (odd)  == n (odd) , k (even)",
        "m (odd)  == n (odd) , k (odd)",
        "m (odd)  != n (odd) , k (even)",
        "m (odd)  != n (odd) , k (odd)",
    ]
    ms = [4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
    ns = [4, 4, 6, 6, 5, 5, 4, 4, 5, 5, 3, 3]
    ks = [2, 3] * 6

    def get_output_shape_from_square_kernel(in_shape: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
        m_, n_ = in_shape[0], in_shape[1]
        out_m = m_ - 1 - (kernel_size % 2)
        out_n = n_ - 1 - (kernel_size % 2)

        return out_m, out_n

    for m, n, k, test_type in zip(ms, ns, ks, test_types):
        X = np.random.random((m, n))
        W = np.ones((k, k))
        Z = scipy.signal.convolve2d(X, W, mode="valid")

        out_shape = get_output_shape_from_square_kernel(X.shape, k)

        print(f"{test_type}\n"
              f"m: {m}, n: {n}, k: {k}\n"
              f"Z.shape = {Z.shape}\n"
              f"Guessed shape = {out_shape}\n")


def main():
    check_mat_sizes()


if __name__ == '__main__':
    main()
