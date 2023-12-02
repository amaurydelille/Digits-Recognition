#include "neuralnetwork.h"
#include "files.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>

#define NUM_CLASSES 10
#define IMAGES_PER_CLASS 892
#define TOTAL_IMAGES (NUM_CLASSES * IMAGES_PER_CLASS)
#define TESTS 113


int main() {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    initialize_network(nn);

    double** trainset = malloc(sizeof(double*) * TOTAL_IMAGES);
    double targets[TOTAL_IMAGES];

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

    /*for(size_t i = 0; i < TOTAL_IMAGES; i+=100){
        printdigit(trainset[i]);
    }*/

    train(nn, trainset, targets, TOTAL_IMAGES);
    

    int correct_predictions = 0;
    for (int i = 1; i <= TESTS; i++) {
        int predicted = predict(nn, testset[i-1]);
        if (predicted == targetstest[i-1]) {
            correct_predictions++;
        }
        printf("Image %d => Pred = %d, True = %d\n", targetstest[i-1], predicted, targetstest[i-1]);
    }

    printf("Accuracy: %f\n", (double)correct_predictions / TESTS);
    freeNeuralNetwork(nn);

    return 0;

}