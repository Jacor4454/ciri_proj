#include "class_tensor.h"

// shadies and funcs
void add_shadie(float* output, float* a, float* b, long n, long m, long red0, long block, int offset, int step){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] + b[i];
}

void handler(bool* alive, int id, int workers, bool* activate, long* n, long* m, long* k, long* block, float** output, float** a, float** b, std::function<void(float*, float*, float*, long, long, long, long, int, int)> *f){
    while(*alive){
        if(*activate){
            (*f)(*output, *a, *b, *n, *m, *k, *block, id, workers);
            *activate = false;
        }
    }
    return;
}

// static defs
int tensor::workers = 4;
std::vector<std::thread> tensor::threads(0);
bool* tensor::activates = nullptr;
bool tensor::alive = false;
long tensor::n = 1;
long tensor::m = 1;
long tensor::k = 1;
long tensor::block = 1;
float* tensor::o = nullptr;
float* tensor::a = nullptr;
float* tensor::b = nullptr;
std::function<void(float*, float*, float*, long, long, long, long, int, int)> tensor::func = add_shadie;

tensor::tensor(std::vector<int> dims_){
    dims = dims_;
    N = 1;
    for(int n : dims)
        N *= n;
    
    contents = (float*)malloc(N * sizeof(float));

    // logging (printing for now)
    // std::cout << "created tensor of size: " << N << " and dims: [";
    // for(int n : dims)
    //     std::cout << n << ", ";
    // std::cout << "]" << std::endl;
}

tensor::tensor(const tensor& t){
    dims = t.getDims();
    N = t.getN();
    contents = (float*)malloc(N * sizeof(float));
    std::memcpy(contents, t.getContents(), N * sizeof(float));
}

tensor::~tensor(){
    free(contents);
}

// static initialiser
void tensor::initaliseThreads(){
    alive = true;
    
    activates = (bool*)malloc(sizeof(bool) * workers);
    for(int i = 0; i < workers; i++)
        activates[i] = false;
    
    for(int i = 0; i < workers; i++){
        threads.push_back(std::thread(handler, &alive, i, workers, &(activates[i]), &n, &m, &k, &block, &o, &a, &b, &func));
    }
}
void tensor::killThreads(){
    alive = false;

    for(int i = 0; i < workers; i++){
        threads[i].join();
    }
    threads.resize(0);

    free(activates);
}

// handlers
std::vector<int> tensor::getDims() const{return dims;};
long tensor::getN() const{return N;};
float* tensor::getContents() const{return contents;};

// functions
tensor tensor::operator+(const tensor& t){
    if(dims != t.getDims())
        throw std::runtime_error("addition of mismatched dimensions");

    tensor output(dims);

    // data
    o = output.getContents();
    a = contents;
    b = t.getContents();

    // dims
    n = N;
    
    // func
    func = &add_shadie;

    //activate threads
    for(long i = 0; i < workers; i++)
        activates[i] = true;

    // loop till done
    bool all = false;
    while(!all){
        all = true;
        for(int i = 0; i < workers; i++)
            if(activates[i]) 
                all = false;
    }
    
    // return
    return output;
}
tensor tensor::operator*(const tensor& t){
    if(dims != t.getDims())
        throw std::runtime_error("addition of mismatched dimensions");

    tensor output(dims);

    for(long i = 0; i < N; i++)
        output[i] = contents[i] * t[i];
    
    return output;
}
float& tensor::operator[](const int& i_){return contents[i_];};
const float& tensor::operator[](const int& i_) const{return contents[i_];};
bool tensor::operator==(const tensor& t){
    if(t.getDims() != dims)
        return false;
    
    for(int i = 0; i < N; i++)
        if(contents[i] != t.getContents()[i])
            return false;
    
    return true;
}

// print
std::ostream& operator<<(std::ostream& os, const tensor& m){
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

