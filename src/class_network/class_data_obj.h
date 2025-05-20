#ifndef CLASS_NETWORK_INP_H
#define CLASS_NETWORK_INP_H

#include "class_learning_network.h"

class learning_data {
    private:
    int epoch = 1;
    bool rand = false;

    int curr_index = 0;
    int curr_epoch_index = 0;
    int curr_epoch = 0;

    public:
    std::vector<std::vector<tensor>> input;
    std::vector<std::vector<tensor>> correct;
    std::function<bool(const tensor&, const tensor&)> eval = [](const tensor& t, const tensor& c){for(int i = 0; i < t.getN(); i++)if(std::round(t[i]) != c[i])return false; return true;};

    learning_data(std::vector<int> inp, std::vector<int> out, int n, int k = 1);

    void setEpoch(int epoch_);
    void setRand(bool rand_);
    int getEpoch();

    void reset();

    void getNext(std::vector<tensor>*&, std::vector<tensor>*&);
    int getEpochIndex();
    int getIndex();
    int getEpochI();

};


#endif