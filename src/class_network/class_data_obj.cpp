#include "class_data_obj.h"

learning_data::learning_data(std::vector<int> inp, std::vector<int> out, int n, int k):
input(n, std::vector<tensor>(k, tensor(inp))),
correct(n, std::vector<tensor>(k, tensor(out)))
{}

void learning_data::setEpoch(int epoch_){epoch = std::max(1, epoch_);}

void learning_data::setRand(bool rand_){rand = rand_;}

void learning_data::getNext(std::vector<tensor>*& inpp, std::vector<tensor>*& corp){
    if(curr_epoch_index >= input.size()){
        curr_epoch++;
        curr_epoch_index = 0;
    }

    if(curr_epoch >= epoch){
        inpp = nullptr;
        corp = nullptr;
        return;
    }
    
    // The Fisher-Yates Shuffling Algorthm
    // Algorthm Design and Applications
    // QA76.9.A43.G6
    // Page 533
    if(rand && (curr_epoch_index == 0)){
        int rng;
        std::vector<tensor> temp;
        for(int i = 0; i < input.size(); i++){
            rng = std::rand() % input.size();
            
            std::swap(input[i], input[rng]);
            std::swap(correct[i], correct[rng]);
        }
    }

    // set output pointers
    inpp = &input[curr_epoch_index];
    corp = &correct[curr_epoch_index];

    // inc
    curr_epoch_index++;
    curr_index++;
}

int learning_data::getEpoch(){return epoch;}

void learning_data::reset(){
    curr_epoch = 0;
    curr_index = 0;
    curr_epoch_index = 0;
}

int learning_data::getEpochIndex(){return curr_epoch_index;}
int learning_data::getIndex(){return curr_index;}
int learning_data::getEpochI(){return curr_epoch;}
