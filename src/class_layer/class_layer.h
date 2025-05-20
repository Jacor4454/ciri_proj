#ifndef CLASS_ABS_LAYER_H
#define CLASS_ABS_LAYER_H

#include "../class_tensor/class_tensor.h"
#include "./learners/class_base.h"

float inverse_sqrt(float number_);

class BaseLayer{

    public:
    static std::vector<int> makeWeightsDims(std::vector<int> in, std::vector<int> out);
    static std::vector<int> makeHWeightsDims(std::vector<int> out);
    static std::vector<int> loadWeightsDims(std::ifstream& f);
    static std::default_random_engine generator;

    virtual void setAcc(activations::accTypes);
    virtual void setLearners(BaseLearnerSelector*);
    virtual void randomise();
    virtual ~BaseLayer();

    virtual void forward(tensor&, const tensor&);
    virtual void backward(tensor&, const tensor&, const tensor&, const tensor&, const tensor&);
    virtual void learn();
    virtual void clear();

    virtual void save(std::ofstream&);
    virtual void save_checkpoint(std::ofstream&);
    virtual void load_checkpoint(std::ifstream&);
    
    virtual std::string getLayerType();

};

#include "./perceptron/class_perceptron.h"
#include "./recersive/class_recersive.h"

#endif