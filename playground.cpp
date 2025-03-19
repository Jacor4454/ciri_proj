
#include <stdlib.h>
#include <vector>
#include <thread>
#include <bits/stdc++.h>

void mult_M_skip(std::vector<std::vector<int>>* output, std::vector<std::vector<int>>* a, std::vector<std::vector<int>>* b, int n, int m, int k, int offset, int step){
    // checks are performed off thread
    for(int i = 0; i < n; i++){
        for(int j = offset; j < m; j+=step){
            int tot = 0;
            for(int k_ = 0; k_ < k; k_++){
                tot += (*a)[i][k_] * (*b)[k_][j];
            }
            (*output)[i][j] = tot;
        }
    }
}

void handler(bool* alive, int id, int workers, bool* activate, int* n, int* m, int* k, std::vector<std::vector<int>>** output, std::vector<std::vector<int>>** a, std::vector<std::vector<int>>** b, std::function<void(std::vector<std::vector<int>>*, std::vector<std::vector<int>>*, std::vector<std::vector<int>>*, int, int, int, int, int)> *f){
    while(*alive){
        if(*activate){
            (*f)(*output, *a, *b, *n, *m, *k, id, workers);
            *activate = false;
        }
    }
    return;
}


int main(){

    int workers = 32;

    std::vector<std::thread> threads;
    threads.reserve(workers);

    bool* activates = (bool*)malloc(sizeof(bool) * workers);
    for(int i = 0; i < workers; i++)
        activates[i] = false;

    bool alive = true;
    int n = 1;
    int m = 4;
    int k = 3;
    std::vector<std::vector<int>>* output;
    std::vector<std::vector<int>>* a;
    std::vector<std::vector<int>>*b;
    std::function<void(std::vector<std::vector<int>>*, std::vector<std::vector<int>>*, std::vector<std::vector<int>>*, int, int, int, int, int)> f;

    std::cout << "created values" << std::endl;

    for(int i = 0; i < workers; i++){
        threads.push_back(std::thread(handler, &alive, i, workers, &activates[i], &n, &m, &k, &output, &a, &b, &f));
    }

    std::cout << "created threads" << std::endl;

    std::vector<std::vector<int>> real_output(n, std::vector<int>(m));
    std::vector<std::vector<int>> real_a(n, std::vector<int>(k));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < k; j++)
            real_a[i][j] = i*k + j;
    std::vector<std::vector<int>> real_b(k, std::vector<int>(m));
    for(int i = 0; i < k; i++)
        for(int j = 0; j < m; j++)
            real_b[i][j] = i*m + j;

    output = &real_output;
    a = &real_a;
    b = &real_b;
    f = &mult_M_skip;

    long long total = 0;
    int rept = 1;

    std::cout << "created task" << std::endl;

    for(int k = 0; k < rept; k++){
        auto start = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < workers; i++)
            activates[i] = true;

        bool all = false;
        while(!all){
            all = true;
            for(int i = 0; i < workers; i++)
                if(activates[i]) all = false;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        total += duration.count();
    }

    std::cout << workers << " threads took: " << total/rept << " nanoseconds on avg over " << rept << " loops" << std::endl;

    std::cout << "A:" << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < k; j++)
        std::cout << real_a[i][j] << ", ";
        std::cout << std::endl;
    }
    std::cout << "B:" << std::endl;
    for(int i = 0; i < k; i++){
        for(int j = 0; j < m; j++)
            std::cout << real_b[i][j] << ", ";
        std::cout << std::endl;
    }
    std::cout << "O:" << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++)
            std::cout << real_output[i][j] << ", ";
        std::cout << std::endl;
    }
    std::cout << "cleanup" << std::endl;

    alive = false;
    for(int i = 0; i < workers; i++)
        threads[i].join();
        
    std::cout << "done" << std::endl;

    return 0;
}

