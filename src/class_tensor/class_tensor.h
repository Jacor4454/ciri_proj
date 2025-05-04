#ifndef CLASS_TENSOR_H
#define CLASS_TENSOR_H

#include <vector>
#include <cstdlib>
#include <bits/stdc++.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <thread>
#include <math.h>

#include "./activators/activators.h"
#include "./gradients/gradients.h"

#include "gradients/gradients.h"
#include "activators/activators.h"

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
        static float* c;
        static std::function<void(float*, float*, float*, float*, long, long, long, long, int, int)> func;
        static bool created;

        // static initialiser
        static void initaliseThreads();
        static void killThreads();
        static void setDims(long, long, long, long);
        static void setData(float*, float*, float*, float*);
        static void setFunc(std::function<void(float*, float*, float*, float*, long, long, long, long, int, int)>);
        static void doJob();
        static bool isActive();
        static void setWorkers(int workers);
        static int getActiveWorkers();
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
    void cpy(const tensor&);
    void cpy(const std::vector<float>&);
    void normalRnd(std::default_random_engine&, float);
    void xavierRnd(std::default_random_engine&, float, float);
    void set(const float);

    // static initialiser
    static void threads_initaliseThreads();
    static void threads_killThreads();
    static bool threads_isActive();
    static void threads_setWorkers(int workers);
    static int threads_getActiveWorkers();

    // handlers
    std::vector<int> getDims() const;
    long getN() const;
    float* getContents() const;

    // accs
    void activate(activations::accTypes);
    void deactivate(tensor& output, activations::accTypes) const;

    // loss
    float loss(const tensor&, errors::errTypes);
    void gradient(tensor&, const tensor&, errors::errTypes);

    // functions
    void add(tensor&, const tensor&) const;
    void add(tensor&, const float&) const;
    void sMult(tensor&, const tensor&) const;
    void sMult(tensor&, const float&) const;
    void mult(tensor&, const tensor&) const;

    // hybrids
    void addAndMult(tensor&, const tensor&, const tensor&) const;
    void multAndInc(tensor&, const tensor&) const;
    void alphaSub(tensor&, float) const;

    // complex multipliers
    void fastDeMultL(tensor&, const tensor&) const;
    void fastDeMultLInc(tensor&, const tensor&) const;
    void fastDeMultR(tensor&, const tensor&) const;

    //operators
    tensor operator+(const tensor&);
    tensor operator+(const float&);
    tensor operator*(const tensor&);
    tensor operator*(const float&);
    tensor operator^(const tensor&);
    float& operator[](const int&);
    const float& operator[](const int&) const;
    bool operator==(const tensor&);

    // debug
    bool nantest() const;
};

std::ostream& operator<<(std::ostream&, const tensor&);

#endif