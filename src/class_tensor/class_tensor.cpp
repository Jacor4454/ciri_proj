#include "class_tensor.h"

// shadies and funcs
void add_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] + b[i];
}

void s_mult_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] * b[i];
}

void mult_M_skip_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                int tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] = tot;
            }
        }
    }
}

void mult_N_skip_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                int tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] = tot;
            }
        }
    }
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
        throw std::runtime_error("straight multiplication of mismatched dimensions");

    tensor output(dims);

    // data
    o = output.getContents();
    a = contents;
    b = t.getContents();

    // dims
    n = N;
    
    // func
    func = &s_mult_shadie;

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
tensor tensor::operator^(const tensor& t){
    // this is LHS 
    // t is RHS
    // n is this(Y)
    // k is this(X) & t(Y)
    // m is t(X)

    // if x is dims.size()
    // the first x-2 dims must equal
    // then dims[x-2] => y
    // then dims[x-1] => x

    if(dims.size() != t.getDims().size())
        throw std::runtime_error("multiplicaiton of wrong length dimensions");

    if(dims.size() < 2)
        throw std::runtime_error("multiplicaiton must have 2 or more dimensions");

    
    int x = dims.size();
    block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("multiplicaiton of wrong value dimensions");
        block *= dims[i];
    }

    if(dims[x-1] != t.getDims()[x-2])
        throw std::runtime_error("multiplicaiton mismatched K dim");
    
    // dims
    n = dims[x-2];
    m = t.getDims()[x-1];
    k = dims[x-1];

    // make output
    std::vector<int> oDims = dims;
    oDims[x-1] = m;
    tensor output(oDims);

    // data
    o = output.getContents();
    a = contents;
    b = t.getContents();
    
    // func
    if(m > 1)
        func = &mult_M_skip_shadie;
    else 
        func = &mult_N_skip_shadie;

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

