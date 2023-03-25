#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
using namespace std;

// Enumerate tunable parameters for increment function
#define BIAS 0
#define STATE 1
#define INPUT 2

/*
Cell State ----(*)----------------------(+)------------------------------------------------|-------------> (Carries forward)
                |                        |                                                 |
                |                 |-----(*)-----|                                        [tanh]
            [Forget Gate]   [Input Gate][Candidate State]          [Output Gate]----------(*)
                |                 |            |                         |                 |
Hidden State----|-----------------|------------|-------------------------|                 |-------------> (Carries forward)
                +                 +            +                         +                 |
Input-----------|-----------------|------------|-------------------------|                 |
                                                                                           |--------------[Output]-----[Softmax]----*[Prediction]*
*/

// Struct will hold the state of the LSTM unit
struct LSTMData
{
    // Memory values
    vector<double> &_input;        // Input to the network to make a prediction with
    vector<double> &_hidden_state; // Hidden state or "short term memory". Processes only current data
    vector<double> &_cell_state;   // Cell state or "long term memory". Processes data from current and previous iterations
    vector<double> &_output;       // Save the value used to generate the prediction each time it is recalculated

    // Weights and Biases
    vector<double> &_input_weight; // Weights that are multiplied with the input value
    vector<double> &_state_weight; // Weights that are multiplied with the internal state value used in each specific gate
    vector<double> &_bias;         // Bias value to add to the result

    // Store the guaranteed size of the data
    int _batch_size;
};

class LSTMGate
{
private:
    // Class contains a struct to hold the internal data values
    LSTMData _internal_state;

public:
    // Functions for each class to overload
    // Pure virtual functions - no base class defintions (=0)
    virtual void feedforward() = 0;                          // Given some input, recalculate the output
    virtual void backpropogate(vector<double> expected) = 0; // Given the expected value for a particular input, tune the parameters so that the predicted value approaches the expected value

    // General member variable getters/setters
    // Getters
    vector<double> &get_input();
    vector<double> &get_hidden_state();
    vector<double> &get_cell_state();
    vector<double> &get_output();
    vector<double> &get_input_weight();
    vector<double> &get_state_weight();
    vector<double> &get_bias();
    size_t get_batch_size();

    // Weights and biases
    // Allow updating the weights/biases by incremental changes, not setting directly
    void increment_tunable(int feature, int index, double inc_amt);
};

// Overload vector multiplication
template <typename T>
vector<T> operator*(const vector<T> rhs)
{
    // Multiplication element-wise
    for (size_t i = 0; i < rhs.size(); i++)
    {
        this.at(i) = this.at(i) * rhs.at(i);
    }

    // return vector
    return *this;
}

// Helper functions
// fast sigmoid
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}
