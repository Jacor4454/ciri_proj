#ifndef CLASS_TENSOR_H
#define CLASS_TENSOR_H

#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <cstring>

// main tensor definition for CPU bound compute
template<class T> class tensor{
    // variable definition
    private:
    std::vector<int> dims;
    long N;
    T* contents;

    // initialisors
    public:
    tensor(std::vector<int> dims_){
        dims = dims_;
        N = 1;
        for(int n : dims)
            N *= n;
        
        contents = (T*)malloc(N * sizeof(T));
    
        // logging (printing for now)
        std::cout << "created tensor of size: " << N << " and dims: [";
        for(int n : dims)
            std::cout << n << ", ";
        std::cout << "]" << std::endl;
    }
    tensor(const tensor& t){
        dims = t.getDims();
        N = t.getN();
        contents = (T*)malloc(N * sizeof(T));
        std::memcpy(contents, t.getContents(), N * sizeof(T));
    }
    ~tensor(){
        free(contents);
    }

    // handlers
    std::vector<int> getDims() const{return dims;};
    long getN() const{return N;};
    std::vector<T> getContents() const{return contents;};

    // functions
    tensor operator+(const tensor& t){
        if(dims != t.getDims())
            throw std::runtime_error("addition of mismatched dimensions");
    
        tensor output(dims);
    
        for(long i = 0; i < N; i++)
            output[i] = contents[i] + t[i];
        
        return output;
    }
    tensor operator*(const tensor& t){
        if(dims != t.getDims())
            throw std::runtime_error("addition of mismatched dimensions");
    
        tensor output(dims);
    
        for(long i = 0; i < N; i++)
            output[i] = contents[i] * t[i];
        
        return output;
    }
    T& operator[](const int& i_){return contents[i_];};
    const T& operator[](const int& i_) const{return contents[i_];};
};

// print
template<class T> 
std::ostream& operator<<(std::ostream& os, const tensor<T>& m){
    std::cout << "created tensor of size: " << m.getN() << " and dims: [";
    for(int n : m.getDims())
        std::cout << n << ", ";
    std::cout << "]" << std::endl;

    std::cout << "[";
    for(long i = 0; i < m.getN(); i++)
        std::cout << m[i] << ", ";
    std::cout << "]" << std::endl;

    return os;
}

#endif