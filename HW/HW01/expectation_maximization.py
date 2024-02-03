import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional, Tuple


@dataclass
class CoinSequence:
    heads: int
    tails: int
    which_coin: int  # either '1' = coinA or '0' = coinB
    which_coin_prob: float

    def __iter__(self):
        for cnt in [self.heads, self.tails]:
            yield cnt


COIN_SEQUENCES: List[CoinSequence] = [
    CoinSequence(heads=9, tails=1, which_coin=1, which_coin_prob=0.5),
    CoinSequence(heads=5, tails=5, which_coin=0, which_coin_prob=0.5),
    CoinSequence(heads=8, tails=2, which_coin=1, which_coin_prob=0.5),
    CoinSequence(heads=7, tails=3, which_coin=1, which_coin_prob=0.5),
    CoinSequence(heads=4, tails=6, which_coin=0, which_coin_prob=0.5),
]

def conditional_probability_of_heads_and_hidden_coin_given_coin_probabilities(
    num_coin_flips: int,
    num_heads: int,
    which_coin: int,
    which_coin_prob: float,
    theta_a: float,
    theta_b: float,
) -> float:
    """
    p(s_i, h_i = {0, 1} | theta_a = X, theta_b = Y)

    :param num_coin_flips: number of total coin flips (heads or tails) for a coin sequence
    :param num_heads: number of heads for a coin sequence
    :param which_coin: either '1' = coin A or '0' = coin B
    :param which_coin_prob: probability of which coin is chosen, e.g., P(h_i = 1) = P(h_i = 0) = 0.5 if both coins are
        as likely to be randomly chosen when conducting the coin sequence events.
    :param theta_a: expectation of coin A for a given t
    :param theta_b: expectation of coin B for a given t
    :return: conditional probability
    """

    if which_coin not in [0, 1]:
        raise RuntimeError(f"which_coin must be either (0, 1). Got '{which_coin}'.")

    if not (0 <= theta_a <= 1):
        raise RuntimeError(f"theta_a must be between [0, 1]. Got '{theta_a}'.")

    if not (0 <= theta_b <= 1):
        raise RuntimeError(f"theta_b must be between [0, 1]. Got '{theta_b}'.")

    combinations = math.comb(num_coin_flips, num_heads)
    if which_coin == 1:
        prob = (theta_a ** num_heads) * ((1 - theta_a) ** (num_coin_flips - num_heads))
    else:
        prob = (theta_b ** num_heads) * ((1 - theta_b) ** (num_coin_flips - num_heads))

    return prob * which_coin_prob * combinations


def log_conditional_probability_of_heads_and_hidden_coin_given_coin_probabilities(
    num_coin_flips: int,
    num_heads: int,
    which_coin: int,
    which_coin_prob: float,
    theta_a: float,
    theta_b: float,
    hidden_var_expectation: Optional[float] = None,
) -> float:
    """
    E_{H_i | S_i, theta_a_hat, theta_b_hat}[l(theta_a, theta_b | S_i, H_i)]

    :param num_coin_flips: number of total coin flips (heads or tails) for a coin sequence
    :param num_heads: number of heads for a coin sequence
    :param which_coin: either '1' = coin A or '0' = coin B
    :param which_coin_prob: probability of which coin is chosen, e.g., P(h_i = 1) = P(h_i = 0) = 0.5 if both coins are
        as likely to be randomly chosen when conducting the coin sequence events.
    :param theta_a: expectation of coin A for a given t
    :param theta_b: expectation of coin B for a given t
    :param hidden_var_expectation:
    :return: conditional probability
    """

    if which_coin not in [0, 1]:
        raise RuntimeError(f"which_coin must be either (0, 1). Got '{which_coin}'.")

    if not (0 <= theta_a <= 1):
        raise RuntimeError(f"theta_a must be between [0, 1]. Got '{theta_a}'.")

    if not (0 <= theta_b <= 1):
        raise RuntimeError(f"theta_b must be between [0, 1]. Got '{theta_b}'.")

    combinations = math.log(math.comb(num_coin_flips, num_heads))
    if which_coin == 1:
        if hidden_var_expectation is None:
            hidden_var_expectation = 1

        prob = hidden_var_expectation * ((num_heads * math.log(theta_a)) + ((num_coin_flips - num_heads) * math.log(1 - theta_a)))
    else:
        if hidden_var_expectation is None:
            hidden_var_expectation = 0

        prob = (1 - hidden_var_expectation) * ((num_heads * math.log(theta_b)) + ((num_coin_flips - num_heads) * math.log(1 - theta_b)))

    return prob + math.log(which_coin_prob) +  combinations


def expectation_of_hi_of_H_given_heads_thetaA_thetaB(
    num_coin_flips: int,
    num_heads: int,
    coinA_prob: float,
    theta_a: float,
    theta_b: float,
) -> float:
    """
    E_{H | S, theta_a_hat, theta_b_hat}[h_i]

    :param num_coin_flips:
    :param num_heads:
    :param coinA_prob:
    :param theta_a:
    :param theta_b:
    :return:
    """

    numerator = conditional_probability_of_heads_and_hidden_coin_given_coin_probabilities(
        num_coin_flips=num_coin_flips,
        num_heads=num_heads,
        which_coin=1,
        which_coin_prob=coinA_prob,
        theta_a=theta_a,
        theta_b=theta_b,
    )

    denominator = numerator + conditional_probability_of_heads_and_hidden_coin_given_coin_probabilities(
        num_coin_flips=num_coin_flips,
        num_heads=num_heads,
        which_coin=0,
        which_coin_prob=1 - coinA_prob,
        theta_a=theta_a,
        theta_b=theta_b,
    )

    return numerator / denominator


def expectations_from_coin_sequences(
    coin_sequences: Union[CoinSequence, List[CoinSequence]],
    coinA_prob: float,
    theta_a: float,
    theta_b: float,
) -> Union[float, List[float]]:

    return_list: bool = True
    if not isinstance(coin_sequences, list):
        coin_sequences = [coin_sequences]
        return_list = False

    expectations: List[float] = []
    for coin_sequence in coin_sequences:
        expectations.append(
            expectation_of_hi_of_H_given_heads_thetaA_thetaB(
                num_coin_flips=sum(coin_sequence),
                num_heads=coin_sequence.heads,
                coinA_prob=coinA_prob,
                theta_a=theta_a,
                theta_b=theta_b,
            )
        )

    if return_list:
        return expectations

    return expectations[0]


def expectation_of_complete_log_likelihood(
    coin_sequences: List[CoinSequence],
    theta_a: float,
    theta_b: float,
) -> float:
    """
    E_{H | S, theta_a_hat, theta_b_hat}[l(theta_a, theta_b | S, H)]


    :param coin_sequences:
    :param theta_a:
    :param theta_b:
    :return:
    """
    per_sequence_expectation: List[float] = []
    for coin_sequence in coin_sequences:
        expectation = expectations_from_coin_sequences(coin_sequence, coinA_prob=coin_sequence.which_coin_prob, theta_a=theta_a, theta_b=theta_b)
        per_sequence_expectation.append(
            log_conditional_probability_of_heads_and_hidden_coin_given_coin_probabilities(
                num_coin_flips=sum(coin_sequence),
                num_heads=coin_sequence.heads,
                which_coin=coin_sequence.which_coin,
                which_coin_prob=coin_sequence.which_coin_prob,
                theta_a=theta_a,
                theta_b=theta_b,
                hidden_var_expectation=expectation,
            )
        )

    return sum(per_sequence_expectation)



def expectation_maximization(
    coin_sequences: List[CoinSequence],
    coinA_prob: float,
    theta_a: float,
    theta_b: float,
) -> Tuple[float, float]:
    """
    compute expectation of coin A and coin B per coin sequence,
    then maximize E_{H | S, theta_a_hat^(t), theta_b_hat^(t)}[l(theta_a, theta_b | S, H)] = Q(theta_a, theta_b | theta_a_hat^(t), theta_b_hat^(t))
    using maximum likelihood estimation (MLE) to estimate parameters for theta_a & theta_b by
    (theta_a_hat^(t+1), theta_b_hat^(t+1)) = arg max_{theta_a, theta_b} Q(theta_a, theta_b | theta_a_hat^(t), theta_b_hat^(t))

    :param coin_sequences:
    :param coinA_prob:
    :param theta_a:
    :param theta_b:
    :return:
    """

    theta_a_mle_num: float = 0.0
    theta_a_mle_den: float = 0.0
    theta_b_mle_num: float = 0.0
    theta_b_mle_den: float = 0.0
    for coin_sequence in coin_sequences:
        expectation = expectations_from_coin_sequences(coin_sequence, coinA_prob=coinA_prob, theta_a=theta_a, theta_b=theta_b)

        theta_a_mle_num += expectation * coin_sequence.heads
        theta_a_mle_den += (sum(coin_sequence) * expectation)

        theta_b_mle_num += (1 - expectation) * coin_sequence.heads
        theta_b_mle_den += (sum(coin_sequence)) * (1 - expectation)

    theta_a_mle = theta_a_mle_num / theta_a_mle_den
    theta_b_mle = theta_b_mle_num / theta_b_mle_den

    return theta_a_mle, theta_b_mle


def optimize_coin_probs_from_coin_sequences(
    coin_sequences: List[CoinSequence],
    coinA_prob: float,
    initial_theta_a: float,
    initial_theta_b: float,
    th: float = 0.0005,
    n_iters: int = 100,
    return_all_vals: bool = False,
) -> Tuple[Union[List[float], float], Union[List[float], float]]:
    """
    compute expectation of coin A and coin B per coin sequence,
    compute maximization step for each coin probability based off of expectation of coin A and coin B.

    :param coin_sequences:
    :param coinA_prob:
    :param initial_theta_a:
    :param initial_theta_b:
    :param th:
    :param n_iters:
    :param return_all_vals:
    :return:
    """

    theta_a = initial_theta_a
    theta_a_diff = 100
    theta_b = initial_theta_b
    theta_b_diff = 100
    expectation = expectation_of_complete_log_likelihood(coin_sequences=coin_sequences, theta_a=theta_a, theta_b=theta_b)
    print(f"Initial theta_a, theta_b, & E[log(complete_likelihood))]: ({theta_a}, {theta_b}, {expectation:.5f})\n")

    cnt = 0

    if return_all_vals:
        theta_a_hats: List[float] = [theta_a]
        theta_b_hats: List[float] = [theta_b]
        while cnt < n_iters and not (theta_a_diff <= th and theta_b_diff <= th):
            theta_a_hat, theta_b_hat = expectation_maximization(
                coin_sequences=coin_sequences,
                coinA_prob=coinA_prob,
                theta_a=theta_a,
                theta_b=theta_b,
            )
            cnt += 1

            theta_a_hats.append(theta_a_hat)
            theta_a_diff = abs(theta_a - theta_a_hat)
            theta_a = theta_a_hat

            theta_b_hats.append(theta_b_hat)
            theta_b_diff = abs(theta_b - theta_b_hat)
            theta_b = theta_b_hat

            expectation = expectation_of_complete_log_likelihood(coin_sequences, theta_a, theta_b)

            print(f"Step {cnt}:\n"
                  f"\ttheta_a = {theta_a:.5f}\tdiff: {theta_b_diff:.5f}\n"
                  f"\ttheta_b = {theta_b:.5f}\tdiff: {theta_b_diff:.5f}\n"
                  f"\texpectation = {expectation:.5f}\n")
        theta_a = theta_a_hats
        theta_b = theta_b_hats
    else:
        while cnt < n_iters and not (theta_a_diff <= th and theta_b_diff <= th):
            theta_a_hat, theta_b_hat = expectation_maximization(
                coin_sequences=coin_sequences,
                coinA_prob=coinA_prob,
                theta_a=theta_a,
                theta_b=theta_b,
            )
            cnt += 1

            theta_a_diff = abs(theta_a - theta_a_hat)
            theta_a = theta_a_hat

            theta_b_diff = abs(theta_b - theta_b_hat)
            theta_b = theta_b_hat

            expectation = expectation_of_complete_log_likelihood(coin_sequences, theta_a, theta_b)

            print(f"Step {cnt}:\n"
                  f"\ttheta_a = {theta_a:.5f}\tdiff: {theta_b_diff:.5f}\n"
                  f"\ttheta_b = {theta_b:.5f}\tdiff: {theta_b_diff:.5f}\n"
                  f"\texpectation = {expectation:.5f}\n")

    return theta_a, theta_b


def main_exp():
    coinA_prob = 0.5
    theta_a = 0.6
    theta_b = 0.5
    expectations = expectations_from_coin_sequences(
        coin_sequences=COIN_SEQUENCES,
        coinA_prob=coinA_prob,
        theta_a=theta_a,
        theta_b=theta_b,
    )

    print("Expectations per sequence:")
    for idx, expectation in enumerate(expectations):
        print(f"\t{idx + 1}: {expectation:.3f}")

    expectation = expectation_of_complete_log_likelihood(
        coin_sequences=COIN_SEQUENCES,
        coinA_prob=coinA_prob,
        theta_a=theta_a,
        theta_b=theta_b,
    )

    print(f"Expectation of complete log likelihood: '{expectation}'")

    theta_a_hat_t_1, theta_b_hat_t_1 = expectation_maximization(coin_sequences=COIN_SEQUENCES, coinA_prob=coinA_prob, theta_a=theta_a, theta_b=theta_b)
    print(f"Maximization step 1:\n"
          f"\ttheta_a_hat_t_1: {theta_a_hat_t_1:.3f}\n"
          f"\ttheta_b_hat_t_1: {theta_b_hat_t_1:.3f}")


def get_curve_of_theta_a_vs_theta_b_em(coinA_prob: float, out_path: Path):
    import matplotlib.pyplot as plt

    initial_theta_a: float = 0.0001
    initial_theta_b: float = 0.0002
    theta_a_hats, theta_b_hats = optimize_coin_probs_from_coin_sequences(
        coin_sequences=COIN_SEQUENCES,
        coinA_prob=coinA_prob,
        initial_theta_a=initial_theta_a,
        initial_theta_b=initial_theta_b,
        th=0.00001,
        return_all_vals=True,
    )
    theta_a_hats_, theta_b_hats_ = optimize_coin_probs_from_coin_sequences(
        coin_sequences=COIN_SEQUENCES,
        coinA_prob=coinA_prob,
        initial_theta_a=theta_a_hats[-1],
        initial_theta_b=theta_a_hats.copy()[-1] - 0.0001,
        th=0.00001,
        return_all_vals=True,
    )

    fig = plt.figure(figsize=(7, 7))
    plt.plot(theta_a_hats, theta_b_hats, c="red")
    plt.plot(theta_a_hats[0], theta_b_hats[0], marker="o", markeredgecolor="red", markerfacecolor="green", label="1st initial point")
    plt.plot(theta_a_hats_, theta_b_hats_, c="green")
    plt.plot(theta_a_hats_[0], theta_b_hats_[0], marker="o", markeredgecolor="green", markerfacecolor="green", label="2nd initial point")
    plt.xlabel("theta_a")
    plt.ylabel("theta_b")
    plt.title("theta_a vs theta_b")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close(fig)


def main():
    coinA_prob: float = 0.5
    initial_theta_a: float = 0.6
    initial_theta_b: float = 0.5
    theta_a, theta_b = optimize_coin_probs_from_coin_sequences(
        coin_sequences=COIN_SEQUENCES,
        coinA_prob=coinA_prob,
        initial_theta_a=initial_theta_a,
        initial_theta_b=initial_theta_b,
        th=0.00001
    )
    print(f"Final theta_a, theta_b: ({theta_a:.5f}, {theta_b:.5f})")

    # out_path = Path(r"C:\Users\cblim\Documents\NCSU\Courses\ECE763\HW\HW01\theta_a-vs-theta_b.png")
    # get_curve_of_theta_a_vs_theta_b_em(coinA_prob=coinA_prob, out_path=out_path)


if __name__ == '__main__':
    main()
