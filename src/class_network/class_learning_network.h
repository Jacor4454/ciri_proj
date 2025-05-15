#ifndef CLASS_LEARNING_NETWORK_H
#define CLASS_LEARNING_NETWORK_H

#include "../rest/src/http_server/http_server.h"

#include "../class_layer/class_layer.h"
#include "../class_layer/perceptron/class_perceptron.h"
#include "../class_layer/recersive/class_recersive.h"

namespace layers {
    typedef enum {
        perceptron,
        recursive,
    } layerTypes;
}

class inputDefObject {
    private:
    std::vector<int> inputDims;
    public:
    inputDefObject(std::vector<int> i):inputDims(i){}
    std::vector<int> getDims(){return inputDims;}
};
class layerDefObject {
    private:
    std::vector<int> outputDims;
    layers::layerTypes lyrTyp;
    activations::accTypes accTyp;
    BaseLearnerSelector* oppTyp;
    public:
    layerDefObject(std::vector<int> o, layers::layerTypes lyrTyp_, activations::accTypes accTyp_, BaseLearnerSelector* oppTyp_=nullptr):outputDims(o){lyrTyp = lyrTyp_; accTyp = accTyp_; oppTyp = oppTyp_;}
    std::vector<int> getDims(){return outputDims;}
    layers::layerTypes getLyrTyp(){return lyrTyp;}
    activations::accTypes getAccTyp(){return accTyp;}
    BaseLearnerSelector* getOppTyp(){return oppTyp;}
};
class outputDefObject {
    private:
    std::vector<int> outputDims;
    layers::layerTypes lyrTyp;
    activations::accTypes accTyp;
    errors::errTypes errTyp;
    BaseLearnerSelector* oppTyp;
    public:
    outputDefObject(std::vector<int> o, layers::layerTypes lyrTyp_, activations::accTypes accTyp_, errors::errTypes errTyp_=errors::SE, BaseLearnerSelector* oppTyp_=nullptr):outputDims(o){lyrTyp = lyrTyp_; accTyp = accTyp_; errTyp = errTyp_; oppTyp = oppTyp_;}
    std::vector<int> getDims(){return outputDims;}
    layers::layerTypes getLyrTyp(){return lyrTyp;}
    activations::accTypes getAccTyp(){return accTyp;}
    errors::errTypes getErrTyp(){return errTyp;}
    BaseLearnerSelector* getOppTyp(){return oppTyp;}
};

// always 8 chars
#define VERSION "00.00.01"
// always 3 chars
#define handshake "END"
namespace save{
    typedef enum{
        inference,
        checkpoint,
    } savetype;
}

class Network {
    private:
    std::vector<BaseLayer*> layers;
    std::vector<std::vector<int>> dimss;
    int N = 0;
    int currMaxItt, lastItt;
    std::vector<std::vector<tensor>> tss;
    std::vector<tensor> invts;
    errors::errTypes lossType;

    // rest server
    HTTPServer myServ;
    std::thread t;
    bool keepServerAlive;

    void resizeItt(int newCurr);

    public:
    Network(inputDefObject, std::vector<layerDefObject>, outputDefObject, int = 1, int = -1);
    Network(const std::string&, int = 1, int = -1);
    ~Network();
    void forward(const std::vector<tensor>& input);
    std::vector<tensor> getOutput();
    void backward(const std::vector<tensor>& correct);

    void learn(const std::vector<std::vector<tensor>>& input, const std::vector<std::vector<tensor>>& correct, std::function<bool(const tensor&, const tensor&)> eval = [](const tensor& t, const tensor& c){for(int i = 0; i < t.getN(); i++)if(std::round(t[i]) != c[i])return false; return true;});
    std::vector<tensor> inference(const std::vector<tensor>& input);

    void save(const std::string&, const save::savetype = save::inference);
};


#endif