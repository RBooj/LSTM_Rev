#include "LSTMGate.h"

// Forget Gate class extends from LSTM gate general class
class ForgetGate : public LSTMGate
{
private:
    // Remember the output of this gate from the last time the network was recalculated
    double _forget_result;

public:
    // TODO: Any public data
    void feedforward();                          // Given some input (current value of _internal_state._input), recalculate the output
    void backpropogate(vector<double> expected); // Given the expected value for a particular input, tune the parameters so that the predicted value approaches the expected value
};

// test