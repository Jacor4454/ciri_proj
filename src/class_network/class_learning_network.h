#ifndef CLASS_LEARNING_NETWORK_H
#define CLASS_LEARNING_NETWORK_H

#include "../rest/src/http_server/http_server.h"
#include "class_network.h"

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

class learningNetwork {
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
    learningNetwork(inputDefObject, std::vector<layerDefObject>, outputDefObject, int = 1, int = -1);
    learningNetwork(const std::string&, int = 1, int = -1);
    ~learningNetwork();
    void forward(const std::vector<tensor>& input);
    std::vector<tensor> getOutput();
    void backward(const std::vector<tensor>& correct);

    void learn(const std::vector<std::vector<tensor>>& input, const std::vector<std::vector<tensor>>& correct);

    void save(const std::string&, const save::savetype = save::inference);
};


#endif