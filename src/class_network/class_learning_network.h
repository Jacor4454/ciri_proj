#ifndef CLASS_LEARNING_NETWORK_H
#define CLASS_LEARNING_NETWORK_H

#include "../rest/src/http_server/http_server.h"
#include "class_network.h"

class learningNetwork {
    private:
    std::vector<BaseLayer*> layers;
    std::vector<std::vector<int>> dimss;
    int N = 0;
    int currMaxItt, lastItt;
    std::vector<std::vector<tensor>> tss;
    std::vector<tensor> invts;

    // rest server
    HTTPServer myServ;
    std::thread t;
    bool keepServerAlive;

    void resizeItt(int newCurr);

    public:
    learningNetwork(inputDefObject, std::vector<layerDefObject>, outputDefObject, int = 1, int = -1);
    ~learningNetwork();
    void forward(const std::vector<tensor>& input);
    std::vector<tensor> getOutput();
    void backward(const std::vector<tensor>& correct);

    void learn(const std::vector<std::vector<tensor>>& input, const std::vector<std::vector<tensor>>& correct);
};


#endif