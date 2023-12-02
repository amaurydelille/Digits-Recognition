#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neuralnetwork.h"
#include <SDL2/SDL_image.h>

double reLU(double x) {
    if (x > 0)
        return x;
    else
        return 0;
}

double reLU_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double mean_squared_error_loss(int target_class, double output_neurons[], int num_output_neurons) {
    double loss = 0.0;

    for (int i = 0; i < num_output_neurons; i++) {
        double target = (i == target_class) ? 1.0 : 0.0;
        loss += pow(output_neurons[i] - target, 2);
    }

    return loss / num_output_neurons;
}

double random_weight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

//ça j'ai demandé à GPT
double categorical_cross_entropy_loss(int target_class, double output_neurons[], int num_output_neurons) {
    double epsilon = 1e-15;
    double loss = 0.0;

    for (int i = 0; i < num_output_neurons; i++) {
        double y_pred = fmax(epsilon, fmin(1 - epsilon, output_neurons[i]));

        if (i == target_class) 
            loss -= log(y_pred);
    }

    return loss;
}


void softmax(double* x, int n) {
    double max = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max)
            max = x[i];
    }

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i] - max); 
        sum += x[i];
    }
    for (int i = 0; i < n; i++) 
        x[i] /= sum;
}

double xavier_weight(int n_in, int n_out) {
    double variance = 2.0 / (n_in + n_out);
    double standard_deviation = sqrt(variance);
    return ((double)rand() / RAND_MAX) * 2 * standard_deviation - standard_deviation;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1.0 - sigmoid_x);
}

void freeNeuralNetwork(NeuralNetwork* nn){
    free(nn->input);
    free(nn->hidden);
    free(nn->output);
    free(nn->bias_h);
    free(nn->bias_o);

    for(size_t i = 0; i < HIDDEN_NEURONS; i++)
        free(nn->weights_ih[i]);

    for(size_t i = 0; i < OUTPUT_NEURONS; i++)
        free(nn->weights_ho[i]);
    
}

void initialize_network(NeuralNetwork* nn) {

    nn->input = malloc(sizeof(double) * INPUT_NEURONS);
    nn->hidden = malloc(sizeof(double) * HIDDEN_NEURONS);
    nn->output = malloc(sizeof(double) * OUTPUT_NEURONS);
    nn->bias_h = malloc(sizeof(double) * HIDDEN_NEURONS);
    nn->bias_o = malloc(sizeof(double) * OUTPUT_NEURONS);
    nn->weights_ih = malloc(sizeof(double) * HIDDEN_NEURONS);
    nn->weights_ho = malloc(sizeof(double) * OUTPUT_NEURONS);

    for(size_t i = 0; i < HIDDEN_NEURONS; i++) {
        nn->bias_h[i] = random_weight();
        nn->weights_ih[i] = malloc(sizeof(double) * INPUT_NEURONS);
        for(size_t j = 0; j < INPUT_NEURONS; j++)
            nn->weights_ih[i][j] = random_weight();//xavier_weight(INPUT_NEURONS, HIDDEN_NEURONS);
    }

    for(size_t i = 0; i < OUTPUT_NEURONS; i++) {
        nn->bias_o[i] = random_weight();
        nn->weights_ho[i] = malloc(sizeof(double) * HIDDEN_NEURONS);
        for(size_t j = 0; j < HIDDEN_NEURONS; j++)
            nn->weights_ho[i][j] = random_weight(); //xavier_weight(HIDDEN_NEURONS, OUTPUT_NEURONS); 
    }
}

void forward_propagation(NeuralNetwork* nn)
{
    for(size_t i = 0; i < HIDDEN_NEURONS; i++)
    {
        nn->hidden[i] = 0;
        for(size_t j = 0; j < INPUT_NEURONS; j++)
            nn->hidden[i] += nn->weights_ih[i][j] * nn->input[j];
        nn->hidden[i] = sigmoid(nn->hidden[i] + nn->bias_h[i]);
    }

    for(size_t i = 0; i < OUTPUT_NEURONS; i++) {
        nn->output[i] = 0;
        for(size_t j = 0; j < HIDDEN_NEURONS; j++)
            nn->output[i] += nn->weights_ho[i][j] * nn->hidden[j];
        nn->output[i] += nn->bias_o[i];
    }

    softmax(nn->output, OUTPUT_NEURONS);
}

void backpropagation(NeuralNetwork *nn, int target_class) {
    double output_errors[OUTPUT_NEURONS];
    double output_deltas[OUTPUT_NEURONS];

    for(size_t i = 0; i < OUTPUT_NEURONS; i++) {
        double target = (i == target_class) ? 1.0 : 0.0;
        output_errors[i] = nn->output[i] - target;
        output_deltas[i] = output_errors[i] * sigmoid_derivative(nn->output[i]); 

        nn->bias_o[i] -= output_deltas[i] * LEARNING_RATE;
        for(size_t j = 0; j < HIDDEN_NEURONS; j++)
            nn->weights_ho[i][j] -= nn->hidden[j] * output_deltas[i] * LEARNING_RATE;
    }

    double hidden_errors[HIDDEN_NEURONS];
    double hidden_deltas[HIDDEN_NEURONS];

    for (size_t i = 0; i < HIDDEN_NEURONS; i++) {
        hidden_errors[i] = 0;
        for (size_t j = 0; j < OUTPUT_NEURONS; j++) 
            hidden_errors[i] += output_deltas[j] * nn->weights_ho[j][i];

        hidden_deltas[i] = hidden_errors[i] * sigmoid_derivative(nn->hidden[i]);
    }

    for(size_t i = 0; i < HIDDEN_NEURONS; i++) {
        nn->bias_h[i] -= hidden_deltas[i] * LEARNING_RATE;
        for(size_t j = 0; j < INPUT_NEURONS; j++)
            nn->weights_ih[i][j] -= nn->input[j] * hidden_deltas[i] * LEARNING_RATE;
    }
}


void train(NeuralNetwork *nn, double** inputs, double* targets, int num_samples) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0;

        for (int i = 0; i < num_samples; i++) {
            for (int j = 0; j < INPUT_NEURONS; j++) {
                nn->input[j] = inputs[i][j];
            }

            forward_propagation(nn);

            double error = 0;
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                double target = (j == (int)targets[i]) ? 1.0 : 0.0;
                error += pow(nn->output[j] - target, 2);
            }
            error /= OUTPUT_NEURONS;
            total_error += error;

            backpropagation(nn, (int)targets[i]);
        }

        total_error /= num_samples;

        if (epoch % 100 == 0)
            printf("Epoch %d: Error = %f\n", epoch, total_error);
    }
}


int predict(NeuralNetwork* nn, double* input) {
    for (int j = 0; j < INPUT_NEURONS; j++) {
        nn->input[j] = input[j];
    }

    forward_propagation(nn);

    int predicted_class = 0;
    double max_output = nn->output[0];
    for (int i = 1; i < OUTPUT_NEURONS; i++) {
        if (nn->output[i] > max_output) {
            max_output = nn->output[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

