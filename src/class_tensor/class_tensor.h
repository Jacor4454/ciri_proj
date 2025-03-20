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

    static struct threadManager {
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

        // static initialiser
        static void initaliseThreads();
        static void killThreads();
        static void setDims(long, long, long, long);
        static void setData(float*, float*, float*);
        static void setFunc(std::function<void(float*, float*, float*, long, long, long, long, int, int)>);
        static void doJob();
    } threadManager;
    // static thread manager

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
    tensor operator+(const tensor&);
    tensor operator*(const tensor&);
    tensor operator^(const tensor&);
    float& operator[](const int&);
    const float& operator[](const int&) const;
    bool operator==(const tensor&);
};

std::ostream& operator<<(std::ostream&, const tensor&);

#endif