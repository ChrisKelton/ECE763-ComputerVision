#include <stdint.h>
#include <iostream>
#include <string>
#include "utils.h"
#include <math.h>
#include <vector>
#include <tuple>

struct CoinSequence {
    uint16_t heads;
    uint16_t tails;
    bool which_coin;
    float which_coin_prob;

    CoinSequence(uint16_t heads, uint16_t tails, bool which_coin, float which_coin_prob):
        heads(heads),
        tails(tails),
        which_coin(which_coin),
        which_coin_prob(which_coin_prob)
    {}

    uint16_t sum() {return heads + tails;}
};


// CoinSequence COIN_SEQUENCES[5] = {
//     {9, 1, true, 0.5},
//     {5, 5, false, 0.5},
//     {8, 2, true, 0.5},
//     {7, 3, true, 0.5},
//     {4, 6, false, 0.5},
// };


float conditional_probability_of_heads_and_hiiden_coin_given_probabilities(
    uint16_t num_coin_flips,
    uint16_t num_heads,
    bool which_coin,
    float which_coin_prob,
    float theta_a,
    float theta_b
) {
    if (theta_a > 1 || theta_a < 0) {
        throw std::runtime_error("theta_a must be between [0, 1]. Got '" + std::to_string(theta_a) + "'");    
    }

    if (theta_b > 1 || theta_b < 0) {
        throw std::runtime_error("theta_b must be between [0, 1]. Got '" + std::to_string(theta_b) + "'");
    }

    auto combinations = nCr(int(num_coin_flips), int(num_heads));
    float prob;
    if (which_coin) {
        prob = pow(theta_a, num_heads) * pow(1 - theta_a, num_coin_flips - num_heads);
    } else {
        prob = pow(theta_b, num_heads) * pow(1 - theta_b, num_coin_flips - num_heads);
    }

    return prob * which_coin_prob * combinations;
}


float expectation_of_hi_of_H_given_heads_thetaA_thetaB(
    uint16_t num_coin_flips,
    uint16_t num_heads,
    float coinA_prob,
    float theta_a,
    float theta_b
) {
    float numerator = conditional_probability_of_heads_and_hiiden_coin_given_probabilities(
        num_coin_flips, num_heads, true, coinA_prob, theta_a, theta_b
    );

    float denominator = numerator + conditional_probability_of_heads_and_hiiden_coin_given_probabilities(
        num_coin_flips, num_heads, false, 1 - coinA_prob, theta_a, theta_b
    );

    return numerator / denominator;
}


float expectations_from_coin_sequences(
    CoinSequence coin_sequences,
    float coinA_prob,
    float theta_a,
    float theta_b
 ) {
    return expectation_of_hi_of_H_given_heads_thetaA_thetaB(
        coin_sequences.sum(),
        coin_sequences.heads,
        coinA_prob,
        theta_a,
        theta_b
    );
 }

 std::vector<float> expectations_from_coin_sequences(
    std::vector<CoinSequence> coin_sequences,
    float coinA_prob,
    float theta_a,
    float theta_b
 ) {
    std::vector<float> expectations;
    for (auto it = begin(coin_sequences); it != end(coin_sequences); ++it) {
        expectations.push_back(
            expectation_of_hi_of_H_given_heads_thetaA_thetaB(
                it->sum(),
                it->heads,
                coinA_prob,
                theta_a,
                theta_b
            )
        );
    }

    return expectations;
 }


std::tuple<float, float> expectation_maximization(
    std::vector<CoinSequence> coin_sequences,
    float coinA_prob,
    float theta_a,
    float theta_b
) {
    float theta_a_mle_num = 0.0, theta_a_mle_den = 0.0;
    float theta_b_mle_num = 0.0, theta_b_mle_den = 0.0;
    for (int i = 0; i < coin_sequences.size(); i++) {
        auto expectation = expectations_from_coin_sequences(coin_sequences[i], coinA_prob, theta_a, theta_b);

        theta_a_mle_num += expectation * coin_sequences[i].heads;
        theta_a_mle_den += coin_sequences[i].sum() * expectation;

        theta_b_mle_num += (1 - expectation) * coin_sequences[i].heads;
        theta_b_mle_den += coin_sequences[i].sum() * (1 - expectation);
    }

    float theta_a_mle = theta_a_mle_num / theta_a_mle_den;
    float theta_b_mle = theta_b_mle_num / theta_b_mle_den;

    return std::make_tuple(theta_a_mle, theta_b_mle);
}


std::tuple<float, float> optimize_coin_probs_from_coin_sequence(
    std::vector<CoinSequence> coin_sequences,
    float coinA_prob,
    float initial_theta_a,
    float initial_theta_b,
    float th = 0.00001,
    uint16_t n_iters = 100
) {

    float theta_a = initial_theta_a;
    float theta_a_diff = 100.0;
    float theta_b = initial_theta_b;
    float theta_b_diff = 100.0;

    std::cout << "Initial theta_a, theta_b: (" << std::to_string(theta_a) << ", " << std::to_string(theta_b) << ")" << std::endl;

    uint16_t cnt = 0;

    while (cnt < n_iters and !(theta_a_diff <= th && theta_b_diff <= th)) {
        auto [theta_a_hat, theta_b_hat] = expectation_maximization(
            coin_sequences,
            coinA_prob,
            theta_a,
            theta_b
        );
        cnt += 1;

        theta_a_diff = abs(theta_a - theta_a_hat);
        theta_a = theta_a_hat;

        theta_b_diff = abs(theta_b - theta_b_hat);
        theta_b = theta_b_hat;

        std::cout << "Step " << std::to_string(cnt) << ":" << std::endl;
        std::cout << "\ttheta_a = " << std::to_string(theta_a) << " diff: " << std::to_string(theta_a_diff) << std::endl;
        std::cout << "\ttheta_b = " << std::to_string(theta_b) << " diff: " << std::to_string(theta_b_diff) << std::endl;
        
    }

    return {theta_a, theta_b};
}


 int main() {
    float coinA_prob = 0.5;
    float initial_theta_a = 0.6;
    float initial_theta_b = 0.5;

    std::vector<CoinSequence> COIN_SEQUENCES;
    COIN_SEQUENCES.push_back(CoinSequence(9, 1, true, 0.5));
    COIN_SEQUENCES.push_back(CoinSequence(5, 5, false, 0.5));
    COIN_SEQUENCES.push_back(CoinSequence(8, 2, true, 0.5));
    COIN_SEQUENCES.push_back(CoinSequence(7, 3, true, 0.5));
    COIN_SEQUENCES.push_back(CoinSequence(4, 6, false, 0.5));

    auto [theta_a, theta_b] = optimize_coin_probs_from_coin_sequence(
        COIN_SEQUENCES,
        coinA_prob,
        initial_theta_a,
        initial_theta_b,
        0.00000001
    );

    std::cout << "Converged theta_a, theta_b: (" << std::to_string(theta_a) << ", " << std::to_string(theta_b) << ")" << std::endl;

    // auto expectations = expectations_from_coin_sequences(
    //     COIN_SEQUENCES,
    //     coinA_prob,
    //     theta_a,
    //     theta_b
    // );

    // std::cout << "Expectations per sequence:" << std::endl;
    // for (int i = 0; i < expectations.size(); i++) {
    //     std::cout << "\t" << std::to_string(i + 1) << ": " << std::to_string(expectations[i]) << std::endl;
    // }

    return 0;
 }