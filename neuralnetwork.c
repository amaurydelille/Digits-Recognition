#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "neuralnetwork.h"
#include "files.h"

#define INPUTS 784
#define HIDDEN 10
#define OUTPUTS 10
#define LEARNING_RATE 0.01
#define EPOCHS 100
#define SAMPLES 60000

double randn() {
    double u1, u2, w, mult;
    static double x1, x2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return x2;
    }

    do {
        u1 = -1 + ((double) rand() / RAND_MAX) * 2;
        u2 = -1 + ((double) rand() / RAND_MAX) * 2;
        w = pow(u1, 2) + pow(u2, 2);
    } while (w >= 1 || w == 0);

    mult = sqrt((-2 * log(w)) / w);
    x1 = u1 * mult;
    x2 = u2 * mult;

    call = !call;

    return x1;
}

double random_weights(){
    double r = randn() * sqrt(2.0 / 784);
    return r;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

double reLu(double x){
    if (x < 0)
        return 0;
    else
        return x;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

void softmax(NeuralNetwork* nn){
    double max_val = nn->output[0];
    for(size_t i = 1; i < OUTPUTS; i++) {
        if (nn->output[i] > max_val) {
            max_val = nn->output[i];
        }
    }

    double sum = 0.0;
    for(size_t i = 0; i < OUTPUTS; i++) {
        nn->output[i] = exp(nn->output[i] - max_val);
        sum += nn->output[i];
    }

    for(size_t i = 0; i < OUTPUTS; i++) {
        nn->output[i] /= sum;
    }
}

void initialize_neuralnetwork(NeuralNetwork* nn){
    nn->inputs = malloc(sizeof(double) * INPUTS);
    nn->hidden = malloc(sizeof(double) * HIDDEN);
    nn->output = malloc(sizeof(double) * OUTPUTS);
    nn->bias_1 = malloc(sizeof(double) * HIDDEN);
    nn->bias_2 = malloc(sizeof(double) * OUTPUTS);
    nn->weights_1 = malloc(sizeof(double*) * HIDDEN);
    nn->weights_2 = malloc(sizeof(double*) * OUTPUTS);

    nn->dZ2 = malloc(sizeof(double) * OUTPUTS);
    nn->dZ1 = malloc(sizeof(double) * HIDDEN);
    nn->db1 = malloc(sizeof(double) * HIDDEN);
    nn->db2 = malloc(sizeof(double) * OUTPUTS);
    nn->dW2 = malloc(sizeof(double) * OUTPUTS);
    nn->dW1 = malloc(sizeof(double) * HIDDEN);

    for(size_t i = 0; i < HIDDEN; i++) {
        nn->weights_1[i] = malloc(sizeof(double) * INPUTS);
        nn->dW1[i] = malloc(sizeof(double) * INPUTS);
        nn->bias_1[i] = random_weights();
        nn->hidden[i] = random_weights();

        for(size_t j = 0; j < INPUTS; j++)
            nn->weights_1[i][j] = random_weights();
    }

    for(size_t i = 0; i < OUTPUTS; i++) {
        nn->weights_2[i] = malloc(sizeof(double) * HIDDEN);
        nn->dW2[i] = malloc(sizeof(double) * HIDDEN);
        nn->bias_2[i] = random_weights();
        nn->output[i] = random_weights();

        for(size_t j = 0; j < HIDDEN; j++)
            nn->weights_2[i][j] = random_weights();
    }
}

void forward_propagation(NeuralNetwork* nn){
    for(size_t i = 0; i < HIDDEN; i++) {
        nn->hidden[i] = 0;
        for(size_t j = 0; j < INPUTS; j++)
            nn->hidden[i] += nn->inputs[j] * nn->weights_1[i][j];

        nn->hidden[i] = reLu(nn->hidden[i] + nn->bias_1[i]);
    }

    for(size_t i = 0; i < OUTPUTS; i++) {
        nn->output[i] = 0;
        for(size_t j = 0; j < HIDDEN; j++)
            nn->output[i] += nn->hidden[j] * nn->weights_2[i][j];

        nn->output[i] = nn->output[i] + nn->bias_2[i];
    }

    softmax(nn);
}

void back_propagation(NeuralNetwork* nn, int* Y, int i){

    int one_hot_Y[OUTPUTS] = {0};
    one_hot_Y[Y[i]] = 1;

    for(size_t i = 0; i < OUTPUTS; i++)
        nn->dZ2[i] = nn->output[i] - one_hot_Y[i];

    for (size_t i = 0; i < OUTPUTS; i++) {
        for (size_t j = 0; j < HIDDEN; j++) {
            nn->dW2[i][j] = nn->dZ2[i] * nn->hidden[j] * 1/SAMPLES;
        }
        nn->db2[i] = nn->dZ2[i] * 1/SAMPLES;
    }

    for (size_t i = 0; i < HIDDEN; i++) {
        nn->dZ1[i] = 0;
        for (size_t j = 0; j < OUTPUTS; j++)
            nn->dZ1[i] += nn->weights_2[j][i] * nn->dZ2[j]; //maybe [j][i] instead of i j
        nn->dZ1[i] *= relu_derivative(nn->hidden[i]);
    }

    for (size_t i = 0; i < HIDDEN; i++) {
        for (size_t j = 0; j < INPUTS; j++) {
            nn->dW1[i][j] = nn->dZ1[i] * nn->inputs[j];
        }
        nn->db1[i] = nn->dZ1[i];
    }
}

void update_params(NeuralNetwork* nn) {
    for (size_t i = 0; i < HIDDEN; i++) {
        for (size_t j = 0; j < INPUTS; j++)
            nn->weights_1[i][j] -= LEARNING_RATE * nn->dW1[i][j];
        nn->bias_1[i] -= LEARNING_RATE * nn->db1[i];
    }
    
    for (size_t i = 0; i < OUTPUTS; i++) {
        for (size_t j = 0; j < HIDDEN; j++)
            nn->weights_2[i][j] -= LEARNING_RATE * nn->dW2[i][j];
        nn->bias_2[i] -= LEARNING_RATE * nn->db2[i];
    }
}

void gradient_descent(NeuralNetwork* nn, int* Y, double** trainset){
    load_params(nn);
    for(size_t i = 0; i < EPOCHS; i++){
        for(size_t j = 0; j < SAMPLES; j++) {
            nn->inputs = trainset[j];

            printdigit(trainset[j]);
            forward_propagation(nn);
            back_propagation(nn, Y, j);
            update_params(nn);

            if (j % 100 == 0)
                printf("EPOCH %zu\n", j);
        } 
        save_params(nn);
    }
}

int getposmax(double* x){
    int pos = 0;
    double max = x[0];

    for(size_t i = 0; i < OUTPUTS; i++){
        if (x[i] > max){
            max = x[i];
            pos = i;
        }
    }
    return pos;
}

double getmax(double* x){
    double max = x[0];

    for(size_t i = 0; i < OUTPUTS; i++){
        if (x[i] > max)
            max = x[i];
    }
    return max;
}

void save_params(NeuralNetwork* nn) {
    FILE *file_wIH = fopen("wIH.w", "w");
    if (file_wIH == NULL) {
        perror("Erreur lors de l'ouverture du fichier wIH.w");
        return;
    }
    for (size_t i = 0; i < HIDDEN; i++) {
        for (size_t j = 0; j < INPUTS; j++) {
            fprintf(file_wIH, "%lf\n", nn->weights_1[i][j]);
        }
        fprintf(file_wIH, "\n");
    }
    fclose(file_wIH);

    FILE *file_wHO = fopen("wHO.w", "w");
    if (file_wHO == NULL) {
        perror("Erreur lors de l'ouverture du fichier wHO.w");
        return;
    }
    for (size_t i = 0; i < OUTPUTS; i++) {
        for (size_t j = 0; j < HIDDEN; j++) {
            fprintf(file_wHO, "%lf\n", nn->weights_2[i][j]);
        }
        fprintf(file_wHO, "\n");
    }
    fclose(file_wHO);

    FILE *file_bIH = fopen("bIH.b", "w");
    if (file_bIH == NULL) {
        perror("Erreur lors de l'ouverture du fichier bIH.b");
        return;
    }
    for (size_t i = 0; i < HIDDEN; i++) {
        fprintf(file_bIH, "%lf\n", nn->bias_1[i]);
    }
    fclose(file_bIH);

    FILE *file_bHO = fopen("bHO.b", "w");
    if (file_bHO == NULL) {
        perror("Erreur lors de l'ouverture du fichier bHO.b");
        return;
    }
    for (size_t i = 0; i < OUTPUTS; i++) {
        fprintf(file_bHO, "%lf\n", nn->bias_2[i]);
    }
    fclose(file_bHO);

    //printf("Les paramètres ont été sauvegardés avec succès.\n");
}


void load_params(NeuralNetwork* nn) {
    FILE *file_wIH = fopen("wIH.w", "r");
    if (file_wIH == NULL) {
        perror("Erreur lors de l'ouverture du fichier wIH.w");
        return;
    }
    for (size_t i = 0; i < HIDDEN; i++) {
        for (size_t j = 0; j < INPUTS; j++) {
            if (fscanf(file_wIH, "%lf", &nn->weights_1[i][j]) != 1) {
                perror("Erreur de lecture dans le fichier wIH.w");
                fclose(file_wIH);
                return;
            }
        }
    }
    fclose(file_wIH);

    FILE *file_wHO = fopen("wHO.w", "r");
    if (file_wHO == NULL) {
        perror("Erreur lors de l'ouverture du fichier wHO.w");
        return;
    }
    for (size_t i = 0; i < OUTPUTS; i++) {
        for (size_t j = 0; j < HIDDEN; j++) {
            if (fscanf(file_wHO, "%lf", &nn->weights_2[i][j]) != 1) {
                perror("Erreur de lecture dans le fichier wHO.w");
                fclose(file_wHO);
                return;
            }
        }
    }
    fclose(file_wHO);

    FILE *file_bIH = fopen("bIH.b", "r");
    if (file_bIH == NULL) {
        perror("Erreur lors de l'ouverture du fichier bIH.b");
        return;
    }
    for (size_t i = 0; i < HIDDEN; i++) {
        if (fscanf(file_bIH, "%lf", &nn->bias_1[i]) != 1) {
            perror("Erreur de lecture dans le fichier bIH.b");
            fclose(file_bIH);
            return;
        }
    }
    fclose(file_bIH);

    FILE *file_bHO = fopen("bHO.b", "r");
    if (file_bHO == NULL) {
        perror("Erreur lors de l'ouverture du fichier bHO.b");
        return;
    }
    for (size_t i = 0; i < OUTPUTS; i++) {
        if (fscanf(file_bHO, "%lf", &nn->bias_2[i]) != 1) {
            perror("Erreur de lecture dans le fichier bHO.b");
            fclose(file_bHO);
            return;
        }
    }
    fclose(file_bHO);

    printf("Les paramètres ont été chargés avec succès.\n");
}

void predict(NeuralNetwork* nn, double* image){
    nn->inputs = image;

    forward_propagation(nn);
    printf("Prediction : %d - %f\n", getposmax(nn->output), getmax(nn->output));
    for(size_t i = 0; i< OUTPUTS; i++)
        printf("%f\n", nn->output[i]);
}