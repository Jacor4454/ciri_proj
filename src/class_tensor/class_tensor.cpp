#include "class_tensor.h"

#include "shadies.cpp"

void handler(bool* alive, int id, int workers, bool* activate, long* n, long* m, long* k, long* block, float** output, float** a, float** b, float** c, std::function<void(float*, float*, float*, float*, long, long, long, long, int, int)> *f){
    while(*alive){
        if(*activate){
            (*f)(*output, *a, *b, *c, *n, *m, *k, *block, id, workers);
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
float* tensor::threadManager::c = nullptr;
std::function<void(float*, float*, float*, float*, long, long, long, long, int, int)> tensor::threadManager::func = add_shadie;
bool tensor::threadManager::created = false;

// static thread functions
void tensor::threadManager::initaliseThreads(){
    if(created)
        throw std::runtime_error("cannot create threads as they have already been made");

    // set status of the thread handler
    created = true;
    alive = true;
    
    // create activate signals
    activates = (bool*)malloc(sizeof(bool) * workers);
    for(int i = 0; i < workers; i++)
        activates[i] = false;
    
    // allocate/start threads
    for(int i = 0; i < workers; i++)
        threads.push_back(std::thread(handler, &alive, i, workers, &(activates[i]), &n, &m, &k, &block, &o, &a, &b, &c, &func));
}
void tensor::threadManager::killThreads(){
    if(!created)
        throw std::runtime_error("cannot stop threads as they have already been stopped");

    // signal stop
    created = false;
    alive = false;

    // join threads, then resize array
    for(int i = 0; i < workers; i++)
        threads[i].join();
    threads.resize(0);

    // free activate signals
    free(activates);
}
void tensor::threadManager::setDims(long n_, long m_, long k_, long block_){
    // set the dims of the data to process
    n = n_;
    m = m_;
    k = k_;
    block = block_;
}
void tensor::threadManager::setData(float* output_, float* a_, float* b_, float* c_){
    // set the data arrays to process
    o = output_;
    a = a_;
    b = b_;
    c = c_;
}
void tensor::threadManager::setFunc(std::function<void(float*, float*, float*, float*, long, long, long, long, int, int)> func_){
    // set the function to process
    func = func_;
}
void tensor::threadManager::doJob(){
    // check threads started
    if(!created)
        throw std::runtime_error("workers not started");
    // check they are alive
    if(!alive)
        throw std::runtime_error("workers dead?!?! this error should be impossible BTW :)");

    // start threads
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

    // set the number of threads to process data
    workers = workers_;
}
int tensor::threadManager::getActiveWorkers(){return threads.size();}


// create tensor
tensor::tensor(std::vector<int> dims_){
    dims = dims_;
    N = 1;
    for(int n : dims)
        N *= n;
    
    // allocate insternal data
    contents = (float*)malloc(N * sizeof(float));
}
// create tensor by copying another tensor
tensor::tensor(const tensor& t){
    dims = t.getDims();
    N = t.getN();
    
    // allocate insternal data
    contents = (float*)malloc(N * sizeof(float));

    // copy insternal data
    std::memcpy(contents, t.getContents(), N * sizeof(float));
}
tensor::~tensor(){
    free(contents);
}
// manual tensor copy
void tensor::cpy(const tensor& t) {
    dims = t.getDims();
    N = t.getN();
    free(contents);
    contents = (float*)malloc(N * sizeof(float));
    std::memcpy(contents, t.getContents(), N * sizeof(float));
}
// copy tensor to tensor
void tensor::cpy(const std::vector<float>& v) {
    if(v.size() != N)
        throw std::runtime_error("cpy vector wrong size");
    
    for(int i = 0; i < N; i++)
        contents[i] = v[i];
}
// randomisers
void tensor::normalRnd(std::default_random_engine& gen, float SD){
    std::normal_distribution<float> distribution(0.0, SD);
    for(int i = 0; i < N; i++)
        contents[i] = distribution(gen);
}
void tensor::xavierRnd(std::default_random_engine& gen, float min, float max){
    std::normal_distribution<float> distribution(min, max);
    for(int i = 0; i < N; i++)
        contents[i] = distribution(gen);
}
// manual set data
void tensor::set(const float f){
    for(int i = 0; i < N; i++)
        contents[i] = f;
}

// pass throughs
void tensor::threads_initaliseThreads(){threadManager::initaliseThreads();}
void tensor::threads_killThreads(){threadManager::killThreads();}
bool tensor::threads_isActive(){return threadManager::isActive();}
void tensor::threads_setWorkers(int workers){threadManager::setWorkers(workers);}
int tensor::threads_getActiveWorkers(){return threadManager::getActiveWorkers();}

// handlers
std::vector<int> tensor::getDims() const{return dims;}
long tensor::getN() const{return N;}
float* tensor::getContents() const{return contents;}

#include "activators/activators.cpp"
// accs
void tensor::activate(activations::accTypes a){
    // data
    threadManager::setData(nullptr, contents, nullptr, nullptr);

    // dims
    threadManager::setDims(N, 1, 1, 1);

    // func
    switch(a){
        case activations::ReLU:
            threadManager::setFunc(ReLU);
            break;
        case activations::Sigmoid:
            threadManager::setFunc(Sigmoid);
            break;
        case activations::tanh:
            threadManager::setFunc(Tanh);
            break;
        // case activations::softmax:
        //     Softmax(data, N);
        //     break;
        // case activations::leakyReLU:
        //     threadManager::setFunc(Leaky_relu);
        //     break;
        default:
            throw std::runtime_error("activation type not supported yet");
    }
    
    //activate threads
    threadManager::doJob();
}
void tensor::deactivate(tensor& output, activations::accTypes a) const{
    if(dims != output.getDims())
        throw std::runtime_error("deactivate dims do not match");

    // data
    threadManager::setData(output.getContents(), contents, nullptr, nullptr);

    // dims
    threadManager::setDims(N, 1, 1, 1);

    // func
    switch(a){
        case activations::ReLU:
            threadManager::setFunc(deReLU);
            break;
        case activations::Sigmoid:
            threadManager::setFunc(deSigmoid);
            break;
        case activations::tanh:
            threadManager::setFunc(deTanh);
            break;
        // case activations::softmax:
        //     deSoftmax(output, data, N);
        //     break;
        // case activations::leakyReLU:
        //     threadManager::setFunc(deLeaky_relu);
        //     break;
        default:
            throw std::runtime_error("deactivation type not supported yet");
    }
    
    //activate threads
    threadManager::doJob();
}

#include "gradients/gradients.cpp"
// loss
float tensor::loss(const tensor& correct, errors::errTypes e){
    if(correct.getDims() != dims)
        throw std::runtime_error("loss correct of incorrect dims");
    
    // make per-thread output
    std::vector<float> outputs(threadManager::getActiveWorkers(), 0);

    // data
    threadManager::setData(&outputs[0], contents, correct.getContents(), nullptr);

    // dims
    threadManager::setDims(N, 1, 1, 1);

    // func
    switch(e){
        case errors::SE:
            threadManager::setFunc(squaredError);
            break;
        case errors::MSE:
            threadManager::setFunc(meanSquaredError);
            break;
        case errors::CE:
            threadManager::setFunc(crossEntropyError);
            break;
        default:
            throw std::runtime_error("loss type not supported yet");
    }

    //activate threads
    threadManager::doJob();

    return std::reduce(outputs.begin(), outputs.end());
}
void tensor::gradient(tensor& output, const tensor& correct, errors::errTypes e){
    if(output.getDims() != dims)
        throw std::runtime_error("gradient output of incorrect dims");
    if(correct.getDims() != dims)
        throw std::runtime_error("gradient correct of incorrect dims");

    // data
    threadManager::setData(output.getContents(), contents, correct.getContents(), nullptr);

    // dims
    threadManager::setDims(N, 1, 1, 1);

    // func
    switch(e){
        case errors::SE:
            threadManager::setFunc(squaredDiff);
            break;
        case errors::MSE:
            threadManager::setFunc(meanSquaredDiff);
            break;
        case errors::CE:
            threadManager::setFunc(crossEntropyDiff);
            break;
        default:
            throw std::runtime_error("loss type not supported yet");
    }
    
    //activate threads
    threadManager::doJob();
}

// learners
void tensor::adagrad(tensor& output, const tensor& d, float alpha){
    if(dims != d.getDims())
        throw std::runtime_error("adagrad of mismatched dimensions");
    if(dims != output.getDims())
        throw std::runtime_error("adagrad output wrong dimensions");

    // lr = alpha/root(sum(a^2)+0.00000001)
    // a -= lr * da

    // self is a^2
    // d is da
    // output is a

    // data
    threadManager::setData(output.getContents(), d.getContents(), contents, &alpha);

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&adagrad_shadie);

    //activate threads
    threadManager::doJob();
}

void tensor::momentum(tensor& output, const tensor& d, float* alphas){
    if(dims != d.getDims())
        throw std::runtime_error("momentum of mismatched dimensions");
    if(dims != output.getDims())
        throw std::runtime_error("momentum output wrong dimensions");

    // lr = b*a + (1-b)*da
    // a -= lr * alpha

    // self is a^2
    // d is da
    // output is a

    // output = weights
    // a = dweights
    // b = mean_dweights
    // c = {alpha, beta}

    // data
    threadManager::setData(output.getContents(), d.getContents(), contents, alphas);

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&momentum_shadie);

    //activate threads
    threadManager::doJob();
}

void tensor::adam_m(const tensor& d, float* alphas){
    if(dims != d.getDims())
        throw std::runtime_error("adam m of mismatched dimensions");

    // self is m

    // data
    threadManager::setData(nullptr, contents, d.getContents(), alphas);

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&adam_m_shadie);

    //activate threads
    threadManager::doJob();
}

void tensor::adam_v(const tensor& d, float* alphas){
    if(dims != d.getDims())
        throw std::runtime_error("adam m of mismatched dimensions");

    // self is v

    // data
    threadManager::setData(nullptr, contents, d.getContents(), alphas);

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&adam_v_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::adam_c(tensor& output, const tensor& m, float* alphas){
    if(dims != m.getDims())
        throw std::runtime_error("momentum of mismatched dimensions");
    if(dims != output.getDims())
        throw std::runtime_error("momentum output wrong dimensions");

    // self is velocity
    // m is m

    // data
    threadManager::setData(output.getContents(), m.getContents(), contents, alphas);

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&adam_c_shadie);

    //activate threads
    threadManager::doJob();
}

// functions
void tensor::add(tensor& output, const tensor& t) const{
    if(dims != t.getDims())
        throw std::runtime_error("addition of mismatched dimensions");
    if(dims != output.getDims())
        throw std::runtime_error("addition output wrong dimensions");

    // data
    threadManager::setData(output.getContents(), contents, t.getContents(), nullptr);

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
    threadManager::setData(output.getContents(), contents, &fcp, nullptr);

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
    threadManager::setData(output.getContents(), contents, t.getContents(), nullptr);

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
    threadManager::setData(output.getContents(), contents, &fcp, nullptr);

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

    // check all excess dims match
    int x = dims.size();
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    // check reletive dims match
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
    threadManager::setData(output.getContents(), contents, t.getContents(), nullptr);
    
    // func
    if(m > 1)
        threadManager::setFunc(&mult_M_skip_shadie);
    else 
        threadManager::setFunc(&mult_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}

// hybrids
void tensor::addAndMult(tensor& output, const tensor& t, const tensor& b) const{
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
        throw std::runtime_error("add+multiplicaiton of wrong length dimensions");

    if(dims.size() < 2)
        throw std::runtime_error("add+multiplicaiton must have 2 or more dimensions");

    // check all excess dims match
    int x = dims.size();
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("add+multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("add+multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    // check reletive dims match
    if(dims[x-1] != t.getDims()[x-2])
        throw std::runtime_error("add+mult mismatched K dim");
    if(output.getDims()[x-1] != t.getDims()[x-1])
        throw std::runtime_error("add+mult mismatched m dim");
    if(output.getDims()[x-2] != dims[x-2])
        throw std::runtime_error("add+mult mismatched n dim");
    if(output.getDims() != b.getDims())
        throw std::runtime_error("add+mult output and adder dimentions mismatch");
    
    // dims
    int m = t.getDims()[x-1];
    threadManager::setDims(dims[x-2], m, dims[x-1], block);

    // data
    threadManager::setData(output.getContents(), contents, t.getContents(), b.getContents());
    
    // func
    if(m > 1)
        threadManager::setFunc(&multNadd_M_skip_shadie);
    else 
        threadManager::setFunc(&multNadd_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::multAndInc(tensor& output, const tensor& t) const{
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
        throw std::runtime_error("add+multiplicaiton of wrong length dimensions");

    if(dims.size() < 2)
        throw std::runtime_error("add+multiplicaiton must have 2 or more dimensions");

    // check all excess dims match
    int x = dims.size();
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("add+multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("add+multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    // check reletive dims match
    if(dims[x-1] != t.getDims()[x-2])
        throw std::runtime_error("add+mult mismatched K dim");
    if(output.getDims()[x-1] != t.getDims()[x-1])
        throw std::runtime_error("add+mult mismatched m dim");
    if(output.getDims()[x-2] != dims[x-2])
        throw std::runtime_error("add+mult mismatched n dim");

    // dims
    int m = t.getDims()[x-1];
    threadManager::setDims(dims[x-2], m, dims[x-1], block);

    // data
    threadManager::setData(output.getContents(), contents, t.getContents(), nullptr);
    
    // func
    if(m > 1)
        threadManager::setFunc(&multNInc_M_skip_shadie);
    else 
        threadManager::setFunc(&multNInc_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::alphaSub(tensor& output, float delta) const{
    if(dims != output.getDims())
        throw std::runtime_error("addition output wrong dimensions");
    // data
    threadManager::setData(output.getContents(), contents, &delta, nullptr);

    // dims
    threadManager::setDims(N, 1, 1, 1);
    
    // func
    threadManager::setFunc(&alpha_sub);

    //activate threads
    threadManager::doJob();
}

// complex multipliers
void tensor::fastDeMultL(tensor& output, const tensor& t) const{
    if(dims.size() != t.getDims().size())
        throw std::runtime_error("multiplicaiton of wrong length dimensions");

    if(dims.size() < 2)
        throw std::runtime_error("multiplicaiton must have 2 or more dimensions");

    // check all excess dims match
    int x = dims.size();
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    // check reletive dims match
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
    threadManager::setData(output.getContents(), contents, t.getContents(), nullptr);
    
    // func
    if(m > 1)
        threadManager::setFunc(&deMultL_M_skip_shadie);
    else 
        threadManager::setFunc(&deMultL_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::fastDeMultLInc(tensor& output, const tensor& t) const{
    if(dims.size() != t.getDims().size())
        throw std::runtime_error("multiplicaiton of wrong length dimensions");

    if(dims.size() < 2)
        throw std::runtime_error("multiplicaiton must have 2 or more dimensions");

    // check all excess dims match
    int x = dims.size();
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    // check reletive dims match
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
    threadManager::setData(output.getContents(), contents, t.getContents(), nullptr);
    
    // func
    if(m > 1)
        threadManager::setFunc(&deMultLInc_M_skip_shadie);
    else 
        threadManager::setFunc(&deMultLInc_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}
void tensor::fastDeMultR(tensor& output, const tensor& t) const{
    if(dims.size() != t.getDims().size())
        throw std::runtime_error("multiplicaiton of wrong length dimensions");

    if(dims.size() < 2)
        throw std::runtime_error("multiplicaiton must have 2 or more dimensions");

    // check all excess dims match
    int x = dims.size();
    long block = 1;
    for(int i = 0; i < x-2; i++){
        if(dims[i] != t.getDims()[i])
            throw std::runtime_error("multiplicaiton of wrong value dimensions");
        if(dims[i] != output.getDims()[i])
            throw std::runtime_error("multiplicaiton output of wrong value dimensions");
        block *= dims[i];
    }

    // check reletive dims match
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
    threadManager::setData(output.getContents(), contents, t.getContents(), nullptr);
    
    // func
    if(m > 1)
        threadManager::setFunc(&deMultR_M_skip_shadie);
    else 
        threadManager::setFunc(&deMultR_N_skip_shadie);

    //activate threads
    threadManager::doJob();
}

// operators
// add
tensor tensor::operator+(const tensor& t){
    // make output
    tensor output(dims);

    add(output, t);
    
    // return
    return output;
}
// add
tensor tensor::operator+(const float& f){
    // make output
    tensor output(dims);

    add(output, f);
    
    // return
    return output;
}
// straight multiply
tensor tensor::operator*(const tensor& t){
    // make output
    tensor output(dims);

    sMult(output, t);
    
    // return
    return output;
}
// straight multiply
tensor tensor::operator*(const float& f){
    // make output
    tensor output(dims);

    sMult(output, f);
    
    // return
    return output;
}
// matrix multiply
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
// indexers
float& tensor::operator[](const int& i_){return contents[i_];}
const float& tensor::operator[](const int& i_) const{return contents[i_];}
bool tensor::operator==(const tensor& t){
    // check dims match
    if(t.getDims() != dims)
        return false;
    
    // return false if a value does not match
    for(int i = 0; i < N; i++)
        if(contents[i] != t.getContents()[i])
            return false;
    
    return true;
}

// debug
// returns true if any are nan
bool tensor::nantest() const{
    // checks all values are not nan
    for(int i = 0; i < N; i++)
        if(contents[i] != contents[i])
            return true;
    return false;
}

// print
std::ostream& operator<<(std::ostream& os, const tensor& m){
    // print size
    std::cout << "Tensor of size: " << m.getN() << " and dims: [";

    // print dims
    for(int n : m.getDims())
        std::cout << n << ", ";
    std::cout << "]" << std::endl;

    // print data
    std::cout << "[";
    for(long i = 0; i < m.getN(); i++)
        std::cout << m[i] << ", ";
    std::cout << "]" << std::endl;

    return os;
}

