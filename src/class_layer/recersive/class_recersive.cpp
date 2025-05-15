#include "class_recersive.h"

recursive::recursive(std::vector<int> in, std::vector<int> out, activations::accTypes acc_):
    weightsDims(BaseLayer::makeWeightsDims(in, out)),
    hweightsDims(BaseLayer::makeHWeightsDims(out)),
    weights(weightsDims),
    rWeights(hweightsDims),
    bias(out),
    prev(out),
    dInt(out),
    hold(out)
{
    dweight = nullptr;
    drweight = nullptr;
    dbias = nullptr;
    acc = acc_;
    if(acc == activations::Sigmoid || acc == activations::tanh){
        weights.xavierRnd(BaseLayer::generator, -1*inverse_sqrt(in[0]*in[1]), inverse_sqrt(in[0]*in[1]));
        rWeights.xavierRnd(BaseLayer::generator, -1*inverse_sqrt(in[0]*in[1]), inverse_sqrt(in[0]*in[1]));
    } else {
        weights.normalRnd(BaseLayer::generator, sqrt(2.0/(in[0]*in[1])));
        rWeights.normalRnd(BaseLayer::generator, sqrt(2.0/(in[0]*in[1])));
    }
    bias.set(0.0);
    bls = new AlphaLearnerSelector(0.01);
}

recursive::recursive(std::ifstream& f):
    weightsDims(BaseLayer::loadWeightsDims(f)),
    hweightsDims(BaseLayer::loadWeightsDims(f)),
    weights(f),
    rWeights(f),
    bias(f),
    prev(bias.getDims()),
    dInt(bias.getDims()),
    hold(bias.getDims())
{
    dweight = nullptr;
    drweight = nullptr;
    dbias = nullptr;
    f.read(reinterpret_cast<char*>(&acc), sizeof(activations::accTypes));
    bls = new AlphaLearnerSelector(0.01);
}

void recursive::setAcc(activations::accTypes a){acc = a;}

void recursive::setLearners(BaseLearnerSelector* bls_){
    if(bls_ == nullptr)
        return;

    if(dweight != nullptr)
        delete dweight;
    if(drweight != nullptr)
        delete drweight;
    if(dbias != nullptr)
        delete dbias;
    if(bls != nullptr)
        delete bls;

    bls = bls_;
    dweight = bls->construct(&weights);
    drweight = bls->construct(&rWeights);
    dbias = bls->construct(&bias);
}

recursive::~recursive(){
    // delete heap memory
    if(dweight != nullptr)
        delete dweight;
    if(drweight != nullptr)
        delete drweight;
    if(dbias != nullptr)
        delete dbias;
    if(bls != nullptr)
        delete bls;
}

void recursive::forward(tensor& output, const tensor& input){

    // do mx+c
    input.addAndMult(output, weights, bias);

    // add mx
    prev.multAndInc(output, rWeights);
    
    // activate
    output.activate(acc);

    // store for next itt
    prev.cpy(output);
}

void recursive::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& prevoutput, const tensor& output){
    // allocate learning tensors if needed
    if(dweight == nullptr)
        dweight = bls->construct(&weights);
    if(drweight == nullptr)
        drweight = bls->construct(&rWeights);
    if(dbias == nullptr)
        dbias = bls->construct(&bias);

    // deactivate output
    output.deactivate(hold, acc);

    // smultiply deactivated output, dOutput and dInt
    dInt.add(dInt, dOutput);
    hold.sMult(hold, dInt);

    // get dweights
    dweight->backprop(input, hold);

    // get dbias
    dbias->backprop(hold);

    // get drweigths
    drweight->backprop(prevoutput, hold);

    // get dinput and dInternal carry
    hold.fastDeMultR(dInput, weights);
    hold.fastDeMultR(dInt, rWeights);
}

void recursive::learn(){
    if(dweight != nullptr)
        dweight->learn();
    if(drweight != nullptr)
        drweight->learn();
    if(dbias != nullptr)
        dbias->learn();
}

void recursive::clear(){
    if(dweight != nullptr)
        dweight->clear();
    if(drweight != nullptr)
        drweight->clear();
    if(dbias != nullptr)
        dbias->clear();
    prev.set(0.0);
    dInt.set(0.0);
}

void recursive::save(std::ofstream& f){
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

    // write h dims size
    i_cache = hweightsDims.size();
    f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));

    // write h dims
    for(int i : hweightsDims){
        i_cache = i;
        f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));
    }

    // save tensors
    weights.save(f);
    rWeights.save(f);
    bias.save(f);
    
    // write acc
    f.write(reinterpret_cast<const char*>(&acc), sizeof(activations::accTypes));

    // if also checkpoint, store learners and learnerSelectors
    // will be wrapped
}

void recursive::save_checkpoint(std::ofstream& f){
    if(dweight == nullptr)
        dweight = bls->construct(&weights);
    if(drweight == nullptr)
        drweight = bls->construct(&rWeights);
    if(dbias == nullptr)
        dbias = bls->construct(&bias);
    
    dweight->checkpoint(f);
    drweight->checkpoint(f);
    dbias->checkpoint(f);
}

void recursive::load_checkpoint(std::ifstream& f){
    if(dweight != nullptr)
        delete dweight;
    if(drweight != nullptr)
        delete drweight;
    if(dbias != nullptr)
        delete dbias;
    
    dweight = BaseLearnerSelector::load(&weights, f);
    drweight = BaseLearnerSelector::load(&rWeights, f);
    dbias = BaseLearnerSelector::load(&bias, f);
}

std::string recursive::getLayerTypeStat(){return "Recersive";}
std::string recursive::getLayerType(){return getLayerTypeStat();}
