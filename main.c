#include "neuralnetwork.h"
#include "files.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include "MNIST_for_C/mnist.h"

#define NUM_CLASSES 10
#define IMAGES_PER_CLASS 892
#define TOTAL_IMAGES (NUM_CLASSES * IMAGES_PER_CLASS)
#define TESTS 113


int main(){
    srand(time(NULL));
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    initialize_neuralnetwork(nn);
    printf("Initialization done\n");

    for(size_t i = 0; i < OUTPUTS; i++)
        printf("%f\n", nn->output[i]);
    printf("\n");

    load_mnist();
    printdigit(train_image[1]);
    printf("Dataset done\n");
    
    /*forward_propagation(nn);
    for(size_t i = 0; i < OUTPUTS; i++)
        printf("%f\n", nn->output[i]);
    printf("\n");

    back_propagation(nn, train_label, 0);
    for(size_t i = 0; i < OUTPUTS; i++)
        printf("%f\n", nn->output[i]);
    printf("\n");

    update_params(nn);
    for(size_t i = 0; i < OUTPUTS; i++)
        printf("%f\n", nn->output[i]);
    printf("\n");

    forward_propagation(nn);
    for(size_t i = 0; i < OUTPUTS; i++)
        printf("%f\n", nn->output[i]);*/
    
    load_params(nn);

    size_t accuracy = 0;
    for(size_t i = 0; i < EPOCHS; i++){
        
        size_t batch_accuracy = 0;
        for(size_t j = 0; j < SAMPLES; j++) {
            nn->inputs = train_image[j];

            //printdigit(train_image[j]);
            forward_propagation(nn);

            if (getposmax(nn->output) == train_label[j]){
                accuracy++;
                batch_accuracy++;
            }

            back_propagation(nn, train_label, j);
            update_params(nn);

            if (j % 100 == 0) {
                double batch_accuracy_percent = (double)(batch_accuracy*100) / (j+1);
                printf("Batch Accuracy after %zu samples in EPOCH %zu: %.2f%%\n", j + 1, i, batch_accuracy_percent);
                printf("%zu / %zu\n", batch_accuracy, j + 1);
                
            }
        } 
        save_params(nn);
        double epoch_accuracy = (double)accuracy / SAMPLES;
        printf("Accuracy for EPOCH %zu: %f\n", i, epoch_accuracy);
    }
    
    //gradient_descent(nn, train_label, train_image);

    /*

    double** trainset = malloc(sizeof(double*) * TOTAL_IMAGES);
    int targets[TOTAL_IMAGES];

    if (trainset == NULL)
        errx(EXIT_FAILURE, "Memory allocation failed for trainset\n");

    //Là je créé le trainset.
    size_t index = 0;
    for (size_t i = 0; i < NUM_CLASSES; i++) {
        char path[20];
        sprintf(path, "training/%zu/", i);

        for (int j = 1; j <= IMAGES_PER_CLASS; j++) {
            char final[50];
            sprintf(final, "%s%zu_%d.png", path, i, j);
            trainset[index] = path_to_input(final);
            targets[index] = i;
            index++;
        }
    }
    printf("Trainset done\n");
    printf("Gradient descent done\n");



    double** testset = malloc(sizeof(double*) * 113);
    int targetstest[] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,  
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9 
    };

    //La je créé le testset
    index = 0;
    for (size_t i = 0; i < TESTS; i++) {
        char path[20];
        sprintf(path, "testing/test%zu.png", i+1);
        testset[index] = path_to_input(path); 
        index++;
    }

    for(size_t i = 0; i < 7; i++)
        predict(nn, testset[i], targetstest[i]);*/
    
    return 0;

}