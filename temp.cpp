#include <vector>
#include <chrono>
#include <iostream>

void add(std::vector<int>& output, std::vector<int>& a, std::vector<int>& b, int n){
    for(int i = 0; i < n; i++)
        output[i] = a[i] + b[i];
}

int main(){

    int n = 1000000;
    long long total = 0;
    std::vector<int> output(n);
    std::vector<int> a(n);
    std::vector<int> b(n);

    int rept = 300;

    for(int k = 0; k < rept; k++){
        auto start = std::chrono::high_resolution_clock::now();

        add(output, a, b, n);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        total += duration.count();
    }

    std::cout << 1 << " true thread took: " << total/rept << " nanoseconds on avg over " << rept << " loops" << std::endl;

    return 0;
}