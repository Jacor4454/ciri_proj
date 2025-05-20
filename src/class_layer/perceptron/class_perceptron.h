#ifndef CLASS_LAYER_PERCEPTRON_H
#define CLASS_LAYER_PERCEPTRON_H

#include "../class_layer.h"

class perceptron : public BaseLayer{
    private:
    std::vector<int> weightsDims;
    tensor weights;
    tensor bias;
    BaseLearner* dweight;
    BaseLearner* dbias;
    tensor hold;
    activations::accTypes acc;
    BaseLearnerSelector* bls;

    public:
    perceptron(std::vector<int>, std::vector<int>);
    perceptron(std::ifstream&);
    void setAcc(activations::accTypes);
    void setLearners(BaseLearnerSelector*);
    void randomise();
    ~perceptron();

    void forward(tensor&, const tensor&);
    void backward(tensor&, const tensor&, const tensor&, const tensor&, const tensor&);
    void learn();
    void clear();

    void save(std::ofstream&);
    void save_checkpoint(std::ofstream&);
    void load_checkpoint(std::ifstream&);

    static std::string getLayerTypeStat();
    std::string getLayerType();
};

#endif