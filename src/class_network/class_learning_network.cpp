#include "class_learning_network.h"


// void exiting(int i){
//     log::add("program terminating");
//     log::close();
//     exit(0);
// }

void handler(bool* keepAlive, HTTPServer* s){
    while(*keepAlive)
        (*s).handleCon();
}

learningNetwork::learningNetwork(inputDefObject i, std::vector<layerDefObject> ls, outputDefObject o, int maxItt, int port):
myServ("0.0.0.0", port)
{
    // get network size and expand/reserve
    N = ls.size()+1;
    currMaxItt = std::max(maxItt, 1);
    lastItt = 0;
    layers.resize(N, nullptr);
    dimss.resize(N+1);
    tss.resize(currMaxItt+1);
    for(int i = 0; i < currMaxItt+1; i++)
        tss[i].reserve(N+1);

    // assemble network
    dimss[0] = i.getDims();
    for(int i = 0; i < N-1; i++){
        dimss[i+1] = ls[i].getDims();
        layers[i] = network::getLayer(ls[i].getLyrTyp(), dimss[i], dimss[i+1]);
        layers[i]->setAcc(ls[i].getAccTyp());
    }
    dimss[N] = o.getDims();
    layers[N-1] = network::getLayer(o.getLyrTyp(), dimss[N-1], dimss[N]);
    layers[N-1]->setAcc(o.getAccTyp());

    // make tensors for passing
    for(auto& v : dimss)
        for(int i = 0; i < currMaxItt+1; i++)
            tss[i].push_back(tensor(v));
    
    for(int i = 0; i < N+1; i++)
        tss[0][i].set(0.0);

    // make tensors for learning
    invts.reserve(N+1);
    for(auto& v : dimss)
        invts.push_back(tensor(v));
    
    // //^C
    // signal(SIGINT, exiting);
    // //abort()
    // signal(SIGABRT, exiting);
    // //sent by "kill" command
    // signal(SIGTERM, exiting);
    // //^Z
    // signal(SIGTSTP, exiting);

    // start webpage thread
    keepServerAlive = false;
    if(port >= 0){
        keepServerAlive = true;
        t = std::thread(handler, &keepServerAlive, &myServ);
        std::cout << "started webserver at:\n" << myServ.getAddr()D;
    }
}

learningNetwork::~learningNetwork(){
    // stop webpage thread
    if(keepServerAlive){
        keepServerAlive = false;
        t.join();
    }

    log::add("program finishing");
}

void learningNetwork::resizeItt(int newCurr){
    // check input (no throw here)
    if(newCurr <= currMaxItt)
        return;
    
    // get network size and expand/reserve
    tss.resize(newCurr+1);
    for(int i = currMaxItt+1; i < newCurr+1; i++)
        tss[i].reserve(N+1);
    
    // make tensors for passing
    for(auto& v : dimss)
        for(int i = currMaxItt+1; i < newCurr+1; i++)
            tss[i].push_back(tensor(v));

    currMaxItt = newCurr;
}

void learningNetwork::forward(const std::vector<tensor>& input){
    int itts = input.size();
    lastItt = itts;

    if(itts > currMaxItt)
        resizeItt(itts);

    for(int i = 0; i < N; i++)
        layers[i]->clear();

    for(int it = 0; it < input.size(); it++){
        tss[it+1][0].cpy(input[it]);

        for(int i = 0; i < N; i++){
            layers[i]->forward(tss[it+1][i+1], tss[it+1][i]);
        }
    }
}

std::vector<tensor> learningNetwork::getOutput(){
    std::vector<tensor> output(lastItt, tensor({}));

    for(int i = 0; i < lastItt; i++)
        output[i].cpy(tss[i+1][N]);

    return output;
}

void learningNetwork::backward(const std::vector<tensor>& correct){
    int itts = correct.size();

    if(itts != lastItt)
        // will be a resizer but for now is throw
        throw std::runtime_error("backward: itteration does not match previous itteration");

    // zero layers carry (if applicable)

    // itterated backward through time
    float loss = 0;
    for(int it = itts-1; it >= 0; it--){
        // differentiate the top
        loss += tss[it+1][N].loss(correct[it], errors::MSE);
        tss[it+1][N].gradient(invts[N], correct[it], errors::MSE);

        for(int i = N-1; i >= 0; i--)
            layers[i]->backward(invts[i], tss[it+1][i], invts[i+1], tss[it][i+1], tss[it+1][i+1]);
    }
    if(loss != loss)
        throw std::runtime_error("nan has occured");
    
    // learn all layers
    for(int i = 0; i < N; i++){
        layers[i]->learn();
    }
}

std::string convArrToStr(std::vector<int> data){
    if(data.size() == 0)
        return "[]";

    std::stringstream ss;
    int n = data.size();

    ss << "[";
    for(int i = 0; i < n-1; i++){
        ss << data[i] << ",";
    }
    ss << data[n-1] << "]";

    return ss.str();
}

void learningNetwork::learn(const std::vector<std::vector<tensor>>& input, const std::vector<std::vector<tensor>>& correct){
    if(input.size() != correct.size())
        throw std::runtime_error("input and lable have different ammounts of data");

    int n = input.size();
    std::vector<int> correctPredictions(n/100, 0);

    // setup restAPI
    myServ.addAPI("/", new Responce::File("./src/docs/index.html", ".html"), GET);
    myServ.addAPI("/index.css", new Responce::File("./src/docs/index.css", ".css"), GET);
    
    Responce::JSON* json = new Responce::JSON();
    (*json)["data"] = convArrToStr(correctPredictions);
    (*json)["length"] = std::to_string((int) n/100);
    myServ.addAPI("/data.json", json, GET);

    int per = 0;
    for(int i = 0; i < n; i++){
        forward(input[i]);
        backward(correct[i]);

        std::vector<tensor> output = getOutput();
        bool corr = true;
        for(int k = 0; k < output.size(); k++)
            for(int j = 0; j < output[k].getN(); j++)
                if(std::round(output[k][j]) != correct[i][k][j])
                    corr = false;
        if(corr)
            per++;
        if(i%100 == 99){
            correctPredictions[i/100] = per;
            (*json)["data"] = convArrToStr(correctPredictions);
            per = 0;
        }
    }
}

