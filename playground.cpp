
#include <stdlib.h>
#include <vector>
#include <thread>
#include <bits/stdc++.h>

void add(std::vector<int>* output, std::vector<int>* a, std::vector<int>* b, int n, int offset, int step){
    for(int i = offset; i < n; i += step)
        (*output)[i] = (*a)[i] + (*b)[i];
}

void handler(bool* alive, int id, int workers, bool* activate, int* n, std::vector<int>** output, std::vector<int>** a, std::vector<int>** b, std::function<void(std::vector<int>*, std::vector<int>*, std::vector<int>*, int, int, int)> *f){
    while(*alive){
        if(*activate){
            (*f)(*output, *a, *b, *n, id, workers);
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
    int n = 1000000;
    std::vector<int>* output;
    std::vector<int>* a;
    std::vector<int>*b;
    std::function<void(std::vector<int>*, std::vector<int>*, std::vector<int>*, int, int, int)> f;

    std::cout << "created values" << std::endl;

    for(int i = 0; i < workers; i++){
        threads.push_back(std::thread(handler, &alive, i, workers, &activates[i], &n, &output, &a, &b, &f));
    }

    std::cout << "created threads" << std::endl;

    std::vector<int> real_output(n);
    std::vector<int> real_a(n);
    std::vector<int> real_b(n);

    output = &real_output;
    a = &real_a;
    b = &real_b;
    f = &add;

    long long total = 0;
    int rept = 300;

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

    std::cout << "cleanup" << std::endl;

    alive = false;
    for(int i = 0; i < workers; i++)
        threads[i].join();
        
    std::cout << "done" << std::endl;

    return 0;
}