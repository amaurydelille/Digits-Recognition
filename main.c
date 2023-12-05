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


int main(int argc, char** argv){
    srand(time(NULL));
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    initialize_neuralnetwork(nn);
    printf("Initialization done\n");
    //load_mnist();
    //printf("Dataset done\n");
    load_params(nn);
    printf("Parameters done\n");
   
    /*
    //GRADIENT DESCENT / TRAINING
    for(size_t i = 0; i < 1; i++){
   
        double accuracy = 0.0;
        double batch_accuracy = 0;
        for(size_t j = 0; j < SAMPLES; j++) {
            nn->inputs = train_image[j];

            //printdigit(train_image[j]);
            forward_propagation(nn);

            if (getposmax(nn->output) == train_label[j]){
                accuracy++;
                batch_accuracy++;
                for(size_t i = 0; i < OUTPUTS; i++)
                    printf("%f\n", nn->output[i]);
                printf("TRUE %d\n", train_label[j]);
                
            }

            if (j % 100 == 0){
                printf("################# ACCURACY : %f #################\n", batch_accuracy/(j+1));
                printf("%f / %zu\n", batch_accuracy, j+1);
            }
            back_propagation(nn, train_label, j);
            update_params(nn);
            save_params(nn);
        } 
        
        double epoch_accuracy = (double)accuracy / SAMPLES;
        printf("###################Accuracy for EPOCH %zu: %f\n", i, epoch_accuracy);
    }*/

    //GLOBAL ACCURACY FOR TESTING
    double testaccuracy = 0.0;
    for(size_t i = 0; i < 10000; i++){
        nn->inputs = test_image[i];

        forward_propagation(nn);
        if (getposmax(nn->output) == test_label[i]) 
            testaccuracy++;
        
    }
    testaccuracy /= 10000;
    printf("Test Accuracy %f\n", testaccuracy);

    double* image = path_to_input(argv[1]);
    predict(nn, image);
    
    return 0;

}