#include "class_perceptron.h"

perceptron::perceptron(std::vector<int> in, std::vector<int> out, activations::accTypes acc_):
    weightsDims(BaseLayer::makeWeightsDims(in, out)),
    weights(weightsDims),
    bias(out),
    hold(out)
{   
    // we want to only define if learning yk
    dweight = nullptr;
    dbias = nullptr;
    acc = acc_;
    if(acc == activations::Sigmoid || acc == activations::tanh)
        weights.xavierRnd(BaseLayer::generator, -1*inverse_sqrt(in[0]*in[1]), inverse_sqrt(in[0]*in[1]));
    else
        weights.normalRnd(BaseLayer::generator, sqrt(2.0/(in[0]*in[1])));
    bias.set(0.0);
    bls = new AlphaLearnerSelector(0.01);
}

perceptron::perceptron(std::ifstream& f):
    weightsDims(BaseLayer::loadWeightsDims(f)),
    weights(f),
    bias(f),
    hold(bias.getDims())
{
    dweight = nullptr;
    dbias = nullptr;
    f.read(reinterpret_cast<char*>(&acc), sizeof(activations::accTypes));
    bls = new AlphaLearnerSelector(0.01);
}


void perceptron::setAcc(activations::accTypes a){acc = a;}

void perceptron::setLearners(BaseLearnerSelector* bls_){
    if(bls_ == nullptr)
        return;
    
    if(dweight != nullptr)
        delete dweight;
    if(dbias != nullptr)
        delete dbias;
    if(bls != nullptr)
        delete bls;

    bls = bls_;
    dweight = bls->construct(&weights);
    dbias = bls->construct(&bias);
}

perceptron::~perceptron(){
    // delete heap memory
    if(dweight != nullptr)
        delete dweight;
    if(dbias != nullptr)
        delete dbias;
    if(bls != nullptr)
        delete bls;
}

void perceptron::forward(tensor& output, const tensor& input){
    // do mx+c
    input.addAndMult(output, weights, bias);

    // activate
    output.activate(acc);
}

void perceptron::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& IGNORE, const tensor& output){
    // allocate learning tensors if needed
    if(dweight == nullptr)
        dweight = bls->construct(&weights);
    if(dbias == nullptr)
        dbias = bls->construct(&bias);

    // deactivate output
    output.deactivate(hold, acc);
    
    // smultiply deactivated output, dOutput and dInt
    hold.sMult(hold, dOutput);

    // get dweights
    dweight->backprop(input, hold);

    // get dbias
    dbias->backprop(hold);

    // get dinput and dInternal carry
    hold.fastDeMultR(dInput, weights);
}

void perceptron::learn(){
    if(dweight != nullptr)
        dweight->learn();
    if(dbias != nullptr)
        dbias->learn();
}

void perceptron::clear(){
    if(dweight != nullptr)
        dweight->clear();
    if(dbias != nullptr)
        dbias->clear();
}

void perceptron::save(std::ofstream& f){
    // write layer type
    std::string name = getLayerType();
    int i_cache = name.size();
    f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));
    f.write(name.c_str(), sizeof(char)*i_cache);

    // write dims size
    i_cache = weightsDims.size();
    f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));

    // write dims
    for(int i : weightsDims){
        i_cache = i;
        f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));
    }

    // save tensors
    weights.save(f);
    bias.save(f);

    // write acc
    f.write(reinterpret_cast<const char*>(&acc), sizeof(activations::accTypes));

    // if also checkpoint, store learners and learnerSelectors
    // will be wrapped
}

void perceptron::save_checkpoint(std::ofstream& f){
    if(dweight == nullptr)
        dweight = bls->construct(&weights);
    if(dbias == nullptr)
        dbias = bls->construct(&bias);
    
    dweight->checkpoint(f);
    dbias->checkpoint(f);
}

void perceptron::load_checkpoint(std::ifstream& f){
    if(dweight != nullptr)
        delete dweight;
    if(dbias != nullptr)
        delete dbias;
    
    dweight = BaseLearnerSelector::load(&weights, f);
    dbias = BaseLearnerSelector::load(&bias, f);
}

std::string perceptron::getLayerTypeStat(){return "Perceptron";}
std::string perceptron::getLayerType(){return getLayerTypeStat();}
