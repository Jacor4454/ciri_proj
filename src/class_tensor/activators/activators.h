#ifndef CLASS_ACTIVATOR_H
#define CLASS_ACTIVATOR_H

#include <math.h>
#include <stdexcept>

// acc lables
namespace activations {
    typedef enum{
        ReLU,
        Sigmoid,
        tanh,
        // softmax,
        leakyReLU,
    } accTypes;
}

#endif