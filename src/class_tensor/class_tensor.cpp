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

void add_K_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] + (*b);
}

void s_mult_K_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] * (*b);
}

void mult_M_skip_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
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
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] = tot;
            }
        }
    }
}

void deMultL_M_skip_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // y == t.y
    // output.x = t.x
    // output.y = x

    // a = kxn tb = kxm

    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*k + k_)*n + i] * b[(blk*k + k_)*m + j];
                }
                output[(blk*n + i)*m + j] = tot;
            }
        }
    }
}

void deMultL_N_skip_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // y == t.y
    // output.x = t.x
    // output.y = x

    // a = kxn tb = kxm

    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*k + k_)*n + i] * b[(blk*k + k_)*m + j];
                }
                output[(blk*n + i)*m + j] = tot;
            }
        }
    }
}

void deMultR_M_skip_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // x == t.x
    // output.x = t.y
    // output.y = y

    // a = nxk tb = mxk

    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*n + i)*k + k_] * b[(blk*m + j)*k + k_];
                }
                output[(blk*n + i)*m + j] = tot;
            }
        }
    }
}

void deMultR_N_skip_shadie(float* output, float* a, float* b, long n, long m, long k, long block, int offset, int step){
    // x == t.x
    // output.x = t.y
    // output.y = y

    // a = nxk tb = mxk

    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*n + i)*k + k_] * b[(blk*m + j)*k + k_];
                }
                output[(blk*n + i)*m + j] = tot;
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

// thread Manager
// static defs
int tensor::threadManager::workers = 1;
std::vector<std::thread> tensor::threadManager::threads(0);
bool* tensor::threadManager::activates = nullptr;
bool tensor::threadManager::alive = false;
long tensor::threadManager::n = 1;
long tensor::threadManager::m = 1;
long tensor::threadManager::k = 1;
long tensor::threadManager::block = 1;
float* tensor::threadManager::o = nullptr;
float* tensor::threadManager::a = nullptr;
float* tensor::threadManager::b = nullptr;
std::function<void(float*, float*, float*, long, long, long, long, int, int)> tensor::threadManager::func = add_shadie;
bool tensor::threadManager::created = false;

// static thread functions
void tensor::threadManager::initaliseThreads(){
    if(created)
        throw std::runtime_error("cannot create threads as they have already been made");

    created = true;
    alive = true;
    
    activates = (bool*)malloc(sizeof(bool) * workers);
    for(int i = 0; i < workers; i++)
        activates[i] = false;
    
    for(int i = 0; i < workers; i++){
        threads.push_back(std::thread(handler, &alive, i, workers, &(activates[i]), &n, &m, &k, &block, &o, &a, &b, &func));
    }
}
void tensor::threadManager::killThreads(){
    if(!created)
        throw std::runtime_error("cannot stop threads as they have already been stopped");

    created = false;
    alive = false;

    for(int i = 0; i < workers; i++){
        threads[i].join();
    }
    threads.resize(0);

    free(activates);
}
void tensor::threadManager::setDims(long n_, long m_, long k_, long block_){
    n = n_;
    m = m_;
    k = k_;
    block = block_;
}
void tensor::threadManager::setData(float* output_, float* a_, float* b_){
    o = output_;
    a = a_;
    b = b_;
}
void tensor::threadManager::setFunc(std::function<void(float*, float*, float*, long, long, long, long, int, int)> func_){
    func = func_;
}
void tensor::threadManager::doJob(){
    if(!created)
        throw std::runtime_error("workers not started");
    
    if(!alive)
        throw std::runtime_error("workers dead?!?! this error should be impossible BTW :)");

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
}
bool tensor::threadManager::isActive(){return created;}
void tensor::threadManager::setWorkers(int workers_){
    if(created)
        throw std::runtime_error("cannot change thread count they have been made");

    workers = workers_;
}
int tensor::threadManager::getActiveWorkers(){return threads.size();}



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

void tensor::threads_initaliseThreads(){threadManager::initaliseThreads();}
void tensor::threads_killThreads(){threadManager::killThreads();}
bool tensor::threads_isActive(){return threadManager::isActive();}
void tensor::threads_setWorkers(int workers){threadManager::setWorkers(workers);}
int tensor::threads_getActiveWorkers(){return threadManager::getActiveWorkers();}

// handlers
std::vector<int> tensor::getDims() const{return dims;}
long tensor::getN() const{return N;}
float* tensor::getContents() const{return contents;}

// functions
void tensor::add(tensor& output, const tensor& t) const{
    if(dims != t.getDims())
        throw std::runtime_error("addition of mismatched dimensions");
    if(dims != output.getDims())
        throw std::runtime_error("addition output wrong dimensions");

    // data
    threadManager::setData(output.getContents(), contents, t.getContents());

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&add_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::add(tensor& output, const float& f) const{
    if(dims != output.getDims())
        throw std::runtime_error("addition output wrong dimensions");

    // data
    float fcp = f;
    threadManager::setData(output.getContents(), contents, &fcp);

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&add_K_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::sMult(tensor& output, const tensor& t) const{
    if(dims != t.getDims())
        throw std::runtime_error("straight multiplication of mismatched dimensions");
    if(dims != output.getDims())
        throw std::runtime_error("straight multiplication output wrong dimensions");

    // data
    threadManager::setData(output.getContents(), contents, t.getContents());

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&s_mult_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::sMult(tensor& output, const float& f) const{
    if(dims != output.getDims())
        throw std::runtime_error("straight multiplication output wrong dimensions");

    // data
    float fcp = f;
    threadManager::setData(output.getContents(), contents, &fcp);

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&s_mult_K_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::mult(tensor& output, const tensor& t) const{
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
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    if(dims[x-1] != t.getDims()[x-2])
        throw std::runtime_error("multiplicaiton mismatched K dim");
    if(output.getDims()[x-1] != t.getDims()[x-1])
        throw std::runtime_error("multiplicaiton mismatched m dim");
    if(output.getDims()[x-2] != dims[x-2])
        throw std::runtime_error("multiplicaiton mismatched n dim");
    
    // dims
    int m = t.getDims()[x-1];
    threadManager::setDims(dims[x-2], m, dims[x-1], block);

    // data
    threadManager::setData(output.getContents(), contents, t.getContents());
    
    // func
    if(m > 1)
        threadManager::setFunc(&mult_M_skip_shadie);
    else 
        threadManager::setFunc(&mult_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}

// complex multipliers
void tensor::fastDeMultL(tensor& output, const tensor& t) const{
    if(dims.size() != t.getDims().size())
        throw std::runtime_error("multiplicaiton of wrong length dimensions");

    if(dims.size() < 2)
        throw std::runtime_error("multiplicaiton must have 2 or more dimensions");

    
    int x = dims.size();
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    // y = dims[x-2]
    if(dims[x-2] != t.getDims()[x-2])
        throw std::runtime_error("multiplicaiton mismatched K dim");
    if(output.getDims()[x-1] != t.getDims()[x-1])
        throw std::runtime_error("multiplicaiton mismatched m dim");
    if(output.getDims()[x-2] != dims[x-1])
        throw std::runtime_error("multiplicaiton mismatched n dim");
    
    // dims
    // y == t.y
    // output.x = t.x
    // output.y = x

    // a = kxn tb = kxm
    int m = t.getDims()[x-1];
    threadManager::setDims(dims[x-1], m, dims[x-2], block);

    // data
    threadManager::setData(output.getContents(), contents, t.getContents());
    
    // func
    if(m > 1)
        threadManager::setFunc(&deMultL_M_skip_shadie);
    else 
        threadManager::setFunc(&deMultL_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::fastDeMultR(tensor& output, const tensor& t) const{
    if(dims.size() != t.getDims().size())
        throw std::runtime_error("multiplicaiton of wrong length dimensions");

    if(dims.size() < 2)
        throw std::runtime_error("multiplicaiton must have 2 or more dimensions");

    
    int x = dims.size();
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    // y = dims[x-2]
    if(dims[x-1] != t.getDims()[x-1])
        throw std::runtime_error("multiplicaiton mismatched K dim");
    if(output.getDims()[x-1] != t.getDims()[x-2])
        throw std::runtime_error("multiplicaiton mismatched m dim");
    if(output.getDims()[x-2] != dims[x-2])
        throw std::runtime_error("multiplicaiton mismatched n dim");
    
    // dims
    // y == t.y
    // output.x = t.x
    // output.y = x

    // a = kxn tb = kxm
    int m = t.getDims()[x-2];
    threadManager::setDims(dims[x-2], m, dims[x-1], block);

    // data
    threadManager::setData(output.getContents(), contents, t.getContents());
    
    // func
    if(m > 1)
        threadManager::setFunc(&deMultR_M_skip_shadie);
    else 
        threadManager::setFunc(&deMultR_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}

// operators
tensor tensor::operator+(const tensor& t){
    tensor output(dims);

    add(output, t);
    
    // return
    return output;
}
tensor tensor::operator+(const float& f){
    tensor output(dims);

    add(output, f);
    
    // return
    return output;
}
tensor tensor::operator*(const tensor& t){
    tensor output(dims);

    sMult(output, t);
    
    // return
    return output;
}
tensor tensor::operator*(const float& f){
    tensor output(dims);

    sMult(output, f);
    
    // return
    return output;
}
tensor tensor::operator^(const tensor& t){
    int x = dims.size();
    int m = t.getDims()[x-1];

    // make output
    std::vector<int> oDims = dims;
    oDims[x-1] = m;
    tensor output(oDims);

    mult(output, t);
    
    // return
    return output;
}
float& tensor::operator[](const int& i_){return contents[i_];}
const float& tensor::operator[](const int& i_) const{return contents[i_];}
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

