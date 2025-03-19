#ifndef CLASS_TENSOR_H
#define CLASS_TENSOR_H

#include <vector>
#include <cstdlib>
#include <bits/stdc++.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <thread>


// main tensor definition for CPU bound compute
class tensor{
    // variable definition
    private:
    // static thread manager
    static int workers;
    static std::vector<std::thread> threads;
    static bool* activates;
    static bool alive;
    static long n;
    static long m;
    static long k;
    static long block;
    static float* o;
    static float* a;
    static float* b;
    static std::function<void(float*, float*, float*, long, long, long, long, int, int)> func;

    // local data
    std::vector<int> dims;
    long N;
    float* contents;

    // initialisors
    public:
    tensor(std::vector<int> dims_);
    tensor(const tensor& t);
    ~tensor();

    // static initialiser
    static void initaliseThreads();
    static void killThreads();

    // handlers
    std::vector<int> getDims() const;
    long getN() const;
    float* getContents() const;

    // functions
    tensor operator+(const tensor& t);
    tensor operator*(const tensor& t);
    float& operator[](const int& i_);
    const float& operator[](const int& i_) const;
    bool operator==(const tensor& t);
};

std::ostream& operator<<(std::ostream& os, const tensor& m);

#endif