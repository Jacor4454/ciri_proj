#ifndef CLASS_LAYER_RECURSIVE_H
#define CLASS_LAYER_RECURSIVE_H

#include "../class_layer.h"

class recursive : public BaseLayer{
    private:
    std::vector<int> weightsDims;
    std::vector<int> hweightsDims;
    tensor weights;
    tensor rWeights;
    tensor bias;
    BaseLearner* dweight;
    BaseLearner* drweight;
    BaseLearner* dbias;
    tensor hold;
    tensor dInt;
    tensor prev;
    activations::accTypes acc;
    BaseLearnerSelector* bls;
    

    public:
    recursive(std::vector<int>, std::vector<int>);
    recursive(std::ifstream&);
    void setAcc(activations::accTypes);
    void setLearners(BaseLearnerSelector*);
    ~recursive();

    void forward(tensor&, const tensor&);
    void backward(tensor&, const tensor&, const tensor&, const tensor&,  const tensor&);
    void learn();
    void clear();

    void save(std::ofstream&);
    void save_checkpoint(std::ofstream&);
    void load_checkpoint(std::ifstream&);

    static std::string getLayerTypeStat();
    std::string getLayerType();
};

#endif