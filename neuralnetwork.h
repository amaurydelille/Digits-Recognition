#pragma once
#include <err.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <math.h>
#include <time.h>

#define INPUT_NEURONS 784
#define HIDDEN_NEURONS 20
#define OUTPUT_NEURONS 10
#define LEARNING_RATE 0.01
#define EPOCHS 20000

typedef struct {
    double* input;
    double* hidden;
    double* output;
    double** weights_ih;
    double** weights_ho;
    double* bias_h;
    double* bias_o;
} NeuralNetwork;

double reLU(double x);
double xavier_weight(int n_in, int n_out);
double random_weight();
double sigmoid(double x);
double sigmoid_derivative(double x);
void initialize_network(NeuralNetwork* neuralnetwork);
void forward_propagation(NeuralNetwork* nn);
void train(NeuralNetwork *nn, double** inputs, double targets[], int samples);
int predict(NeuralNetwork* nn, double* input);
double categorical_cross_entropy_loss(int target_class, double output_neurons[], int num_output_neurons);
void softmax(double* x, int n);
double reLU_derivative(double x);
void freeNeuralNetwork(NeuralNetwork* nn);
void backpropagation(NeuralNetwork *nn, int target);
double mean_squared_error_loss(int target_class, double output_neurons[], int num_output_neurons);