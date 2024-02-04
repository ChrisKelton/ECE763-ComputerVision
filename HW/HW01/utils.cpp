#include "utils.h"

int fact(int n) {
    if (n == 0) {
        return 1;
    }
    int res = 1;
    for (int i = 2; i <= n; i++) {
        res = res * i;
    }
    return res;
}


int nCr(int n, int r) {
    return fact(n) / (fact(r) * fact(n - r));
}


// int main() {
//     int n = 5, r = 3;
//     std::cout << std::to_string(nCr(n, r));
//     return 0;
// }