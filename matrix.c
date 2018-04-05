#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include "matrix.h"


/*

P-THREADING

Done: 
compute_sum
compute_freq
compute_max
compute_min

sorted
cloned
transposed
reversed
rotated

scalar_add
scalar_multiply

matrix_add
matrix_mul
matrix_pow

construct uniform
construct identity
construct sequence


To do:

compute_trace

*/

static int g_seed = 0;

static ssize_t g_width = 0;
static ssize_t g_height = 0;
static ssize_t g_elements = 0;

static ssize_t g_nthreads = 1;

typedef struct{ 
    size_t id;
    const float* array;
    double result; 
} sum_args;

typedef struct{
    float key;
    size_t id;
    const float* array;
    double freq;
} freq_args;

typedef struct{
    size_t id;
    const float* array;
    double result;
} max_args;

typedef struct{
    size_t id;
    const float* array;
    double result;
} min_args;

typedef struct{
    size_t id;
    const float* array;
    float* result;
    float scalar;
} sAdd_args;

typedef struct{
    size_t id;
    const float* array;
    float* result;
    float scalar;
} sMul_args;

typedef struct{
    size_t id;
    float* result;
    float scalar;
} uniform_args;

typedef struct{
    size_t id;
    float* result;
    const float* matrix;
} rotated_args;

typedef struct{
    size_t id;
    const float* matrix_a;
    const float* matrix_b;
    float* result;
} matrixAdd_args;

typedef struct{
    size_t id;
    const float* matrix_a;
    const float* matrix_b;
    float* result;
} matrixMul_args;

typedef struct{
    size_t id;
    float* result;
} identity_args;

typedef struct{
    size_t id;
    float* result;
    float start;
    float step;
} sequence_args;

typedef struct{
    size_t id;
    float* result;
} sorted1_args;

typedef struct{
    size_t id;
    float* result;
} sorted2_args;

typedef struct{
    size_t id;
    float* result;
} sorted3_args;

typedef struct{
    size_t id;
    float* result;
} sorted4_args;

typedef struct{
    size_t id;
    float* result;
} sorted5_args;

typedef struct{
    size_t id;
    float* result;
} sorted6_args;

typedef struct{
    size_t id;
    float* result;
    const float* matrix;
} cloned_args;

typedef struct{
    size_t id;
    float* result;
    const float* matrix;
} transposed_args;

typedef struct{
    size_t id;
    float* result;
    const float* matrix;
} reversed_args;

////////////////////////////////
///     UTILITY FUNCTIONS    ///
////////////////////////////////

/**
 * Returns pseudorandom number determined by the seed.
 */
int fast_rand(void) {

	g_seed = (214013 * g_seed + 2531011);
	return (g_seed >> 16) & 0x7FFF;
}

/**
 * Sets the seed used when generating pseudorandom numbers.
 */
void set_seed(int seed) {

	g_seed = seed;
}

/**
 * Sets the number of threads available.
 */
void set_nthreads(ssize_t count) {
	g_nthreads = count;
}

/**
 * Sets the dimensions of the matrix.
 */
void set_dimensions(ssize_t order) {

	g_width = order;
	g_height = order;

	g_elements = g_width * g_height;
}

/**
 * Displays given matrix.
 */
void display(const float* matrix) {

	for (ssize_t y = 0; y < g_height; y++) {
		for (ssize_t x = 0; x < g_width; x++) {
			if (x > 0) printf(" ");
			printf("%.2f", matrix[y * g_width + x]);
		}

		printf("\n");
	}
}

/**
 * Displays given matrix row.
 */
void display_row(const float* matrix, ssize_t row) {

	for (ssize_t x = 0; x < g_width; x++) {
		if (x > 0) printf(" ");
		printf("%.2f", matrix[row * g_width + x]);
	}

	printf("\n");
}

/**
 * Displays given matrix column.
 */
void display_column(const float* matrix, ssize_t column) {

	for (ssize_t i = 0; i < g_height; i++) {
		printf("%.2f\n", matrix[i * g_width + column]);
	}
}

/**
 * Displays the value stored at the given element index.
 */
void display_element(const float* matrix, ssize_t row, ssize_t column) {

	printf("%.2f\n", matrix[row * g_width + column]);
}

////////////////////////////////
///   MATRIX INITALISATIONS  ///
////////////////////////////////

/**
 * Returns new matrix with all elements set to zero.
 */
float* new_matrix(void) {

	return calloc(g_elements, sizeof(float));
}

/**
 * Returns new identity matrix.
 */

void* identity_worker(void* args){
    //Copy the args into the sum_args struct
    identity_args* wargs = (identity_args*) args;
    
    int chunk = (g_width / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_width : (wargs->id + 1) * chunk;
    
    //printf("Thread %d is starting at %d and ending at %d\n", (int)wargs->id + 1, (int)start, (int)end);
    
    for (ssize_t i = start; i < end; i++){
        wargs->result[i*g_width + i] = 1;
    }
    
    return NULL;
}

float* identity_matrix(void) {
    
	float* result = new_matrix();
    
    identity_args* args = malloc(sizeof(identity_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (identity_args) {
            .id = i,
            .result = result,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, identity_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }

	for (ssize_t i = 0; i < g_width; i++) {
        result[i + i*g_width] = 1;
	}
    
    free(args);

	return result;
}

/**
 * Returns new matrix with elements generated at random using given seed.
 */
float* random_matrix(int seed) {

	float* result = new_matrix();

	set_seed(seed);

	for (ssize_t i = 0; i < g_elements; i++) {
		result[i] = fast_rand();
	}

	return result;
}

void* uniform_worker(void* args){
    
    //Copy the args into the sum_args struct
    uniform_args* wargs = (uniform_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    float num = wargs->scalar;
    
    for (size_t i = start; i < end; i++){
        wargs->result[i] = num;
    }
    
    return NULL;
}

/**
 * Returns new matrix with all elements set to given value.
 */
float* uniform_matrix(float value) {
    
    float* res = new_matrix();
    
    if (value == 0){
        return res;
    }
    
    uniform_args* args = malloc(sizeof(uniform_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (uniform_args) {
            .id = i,
            .result = res,
            .scalar = value,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, uniform_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    free(args);
    
	return res;
}

void* sequence_worker(void* args){
    //Copy the args into the sum_args struct
    sequence_args* wargs = (sequence_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    //printf("Thread %d is starting at %d and ending at %d\n", (int)wargs->id + 1, (int)start, (int)end);
    
    for (ssize_t i = start; i < end; i++){
        wargs->result[i] = i*wargs->step + wargs->start;
    }
    
    return NULL;
}

/**
 * Returns new matrix with elements in sequence from given start and step
 */
float* sequence_matrix(float start, float step) {

	float* result = new_matrix();
    
    sequence_args* args = malloc(sizeof(sequence_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (sequence_args) {
            .id = i,
            .result = result,
            .start = start,
            .step = step,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, sequence_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    free(args);

	return result;
}

////////////////////////////////
///     MATRIX OPERATIONS    ///
////////////////////////////////


void* cloned_worker(void* args){
    //Copy the args into the sum_args struct
    cloned_args* wargs = (cloned_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    for (size_t i = start; i < end; i++){
        wargs->result[i] = wargs->matrix[i];
    }
    
    return NULL;
}

/**
 * Returns new matrix with elements cloned from given matrix.
 */
float* cloned(const float* matrix) {

	float* result = new_matrix();

    cloned_args* args = malloc(sizeof(cloned_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (cloned_args) {
            .id = i,
            .result = result,
            .matrix = matrix,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, cloned_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }

    free(args);
	return result;
}

int compareNum (const void * a, const void * b){
   return ( *(float*)a - *(float*)b );
}

void* sorted1_worker(void* args){
    //Copy the args into the sum_args struct
    sorted1_args* wargs = (sorted1_args*) args;
    
    int chunk = (g_elements / 2);
    
    qsort(wargs->result, chunk, sizeof(float), compareNum);
    
    return NULL;
}

void* sorted2_worker(void* args){
    sorted2_args* wargs = (sorted2_args*) args;
    
    int chunk = 0;
    
    if (g_elements%2 == 0){
        chunk = (g_elements / 2);
    }
    else{
        chunk = (g_elements / 2) + 1;
    }
    
    qsort(wargs->result, chunk, sizeof(float), compareNum);
    
    return NULL;
}

void* sorted3_worker(void* args){
    //Copy the args into the sum_args struct
    sorted3_args* wargs = (sorted3_args*) args;
    
    int chunk = (g_elements / 4);
    
    qsort(wargs->result, chunk, sizeof(float), compareNum);
    
    return NULL;
}
void* sorted4_worker(void* args){
    //Copy the args into the sum_args struct
    sorted4_args* wargs = (sorted4_args*) args;
    
    int chunk = (g_elements / 4);
    
    qsort(wargs->result, chunk, sizeof(float), compareNum);
    
    return NULL;
}
void* sorted5_worker(void* args){
    //Copy the args into the sum_args struct
    sorted5_args* wargs = (sorted5_args*) args;
    
    int chunk = (g_elements / 4);
    
    qsort(wargs->result, chunk, sizeof(float), compareNum);
    
    return NULL;
}
void* sorted6_worker(void* args){
    sorted6_args* wargs = (sorted6_args*) args;
    
    int chunk = 0;
    
    if (g_elements%2 == 0){
        chunk = (g_elements / 4);
    }
    else{
        chunk = (g_elements / 4) + 1;
    }
    
    qsort(wargs->result, chunk, sizeof(float), compareNum);
    
    return NULL;
}

/**
 * Returns new matrix with elements in ascending order.
 */
float* sorted(const float* matrix) {
    //clock_t start = clock(), diff;
    
    if (g_nthreads > 3){
        
        float* result = calloc(g_elements/4, sizeof(float));
        result = memcpy(result, matrix, ((int)(g_elements/4))*sizeof(float));
        
        float* result2 = calloc(g_elements/4, sizeof(float));
        result2 = memcpy(result2, matrix + g_elements/4, ((int)(g_elements/4))*sizeof(float));
        
        float* result3 = calloc(g_elements/4, sizeof(float));
        result3 = memcpy(result3, matrix + g_elements/2, ((int)(g_elements/4))*sizeof(float));
        
        int length1 = g_elements/4;
        int length2 = 0;
        
        float* result4;
        
        if (g_elements%2 == 0){
            result4 = calloc(g_elements/4, sizeof(float));
            result4 = memcpy(result4, matrix + (int)(3*g_elements/4), g_elements*sizeof(float)/4);
            length2 = g_elements/4;
        }
        else{
            result4 = calloc((int)(g_elements/4) + 1, sizeof(float));
            length2 = g_elements/4 + 1;
            //printf("length2 = %d\n", length2);
            result4 = memcpy(result4, matrix + (int)(3*g_elements/4), length2 * sizeof(float));
        }
        
        sorted3_args* args3 = malloc(sizeof(sorted3_args));
        sorted4_args* args4 = malloc(sizeof(sorted4_args));
        sorted5_args* args5 = malloc(sizeof(sorted5_args));
        sorted6_args* args6 = malloc(sizeof(sorted6_args));

        //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
        args3[0] = (sorted3_args) {
            .id = 0,
            .result = result,
        };
        
        args4[0] = (sorted4_args){
            .id = 1,
            .result = result2,
        };
        
        args5[0] = (sorted5_args) {
            .id = 2,
            .result = result3,
        };
        
        args6[0] = (sorted6_args){
            .id = 3,
            .result = result4,
        };
        
        pthread_t thread_ids[4];
        
        pthread_create(thread_ids, NULL, sorted3_worker, args3);
        pthread_create(thread_ids + 1, NULL, sorted4_worker, args4);
        pthread_create(thread_ids + 2, NULL, sorted5_worker, args5);
        pthread_create(thread_ids + 3, NULL, sorted6_worker, args6);
        
        pthread_join(thread_ids[0], NULL);
        pthread_join(thread_ids[1], NULL);
        pthread_join(thread_ids[2], NULL);
        pthread_join(thread_ids[3], NULL);
        
        /*
        printf("array 1: \n");
        for (int i = 0; i < length1; i++){
            printf(" %f ", args3[0].result[i]);
        }
        printf("\narray 2: \n");
        for (int i = 0; i < length1; i++){
            printf(" %f ", args4[0].result[i]);
        }
        printf("\narray 3: \n");
        for (int i = 0; i < length1; i++){
            printf(" %f ", args5[0].result[i]);
        }
        printf("\narray 4: \n");
        for (int i = 0; i < length2; i++){
            printf(" %f ", args6[0].result[i]);
        }
        */

        
        int i = 0;
        int j = 0;
        int k = 0;
        
        float* firstTwo = new_matrix();
        while (i < length1 && j < length1)
        {
            if (args3[0].result[i] < args4[0].result[j])
            {
                firstTwo[k] = args3[0].result[i];
                //printf("just inputted %f into k = %d\n", toReturn[k], k);
                i++;
            }
            else
            {
                firstTwo[k] = args4[0].result[j];
                //printf("just inputted %f into k = %d\n", toReturn[k], k);
                j++;
            }
            k++;
        }

        while (i < length1)
        {
            firstTwo[k] = args3[0].result[i];
            //printf("just inputted %f into k = %d\n", toReturn[k], k);
            i++;
            k++;
        }

        while (j < length1)
        {
            firstTwo[k] = args4[0].result[j];
            //printf("just inputted %f into k = %d\n", toReturn[k], k);
            j++;
            k++;
        }
        
        i = 0;
        j = 0;
        k = 0;
        
        float* lastTwo = new_matrix();
        while (i < length1 && j < length2)
        {
            if (args5[0].result[i] < args6[0].result[j])
            {
                lastTwo[k] = args5[0].result[i];
                //printf("just inputted %f into k = %d\n", toReturn[k], k);
                i++;
            }
            else
            {
                lastTwo[k] = args6[0].result[j];
                //printf("just inputted %f into k = %d\n", toReturn[k], k);
                j++;
            }
            k++;
        }

        while (i < length1)
        {
            lastTwo[k] = args5[0].result[i];
            //printf("just inputted %f into k = %d\n", toReturn[k], k);
            i++;
            k++;
        }

        while (j < length2)
        {
            lastTwo[k] = args6[0].result[j];
            //printf("just inputted %f into k = %d\n", toReturn[k], k);
            j++;
            k++;
        }
        
        i = 0;
        j = 0;
        k = 0;
        
        float* toReturn = new_matrix();
        while (i < length1 + length1 && j < length1 + length2)
        {
            if (firstTwo[i] < lastTwo[j])
            {
                toReturn[k] = firstTwo[i];
                //printf("just inputted %f into k = %d\n", toReturn[k], k);
                i++;
            }
            else
            {
                toReturn[k] = lastTwo[j];
                //printf("just inputted %f into k = %d\n", toReturn[k], k);
                j++;
            }
            k++;
        }

        while (i < length1 + length1)
        {
            toReturn[k] = firstTwo[i];
            //printf("just inputted %f into k = %d\n", toReturn[k], k);
            i++;
            k++;
        }

        while (j < length1 + length2)
        {
            toReturn[k] = lastTwo[j];
            //printf("just inputted %f into k = %d\n", toReturn[k], k);
            j++;
            k++;
        }
        
        free(result);
        free(result2);
        free(result3);
        free(result4);
        free(firstTwo);
        free(lastTwo);
        free(args3);
        free(args4);
        free(args5);
        free(args6);
        
        //diff = clock() - start;
        //int msec = diff * 1000 / CLOCKS_PER_SEC;
        //printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
        
        return toReturn;
        
    }
    
    else if(g_nthreads > 1){
        float* result = calloc(g_elements/2, sizeof(float));
        result = memcpy(result, matrix, ((int)(g_elements/2))*sizeof(float));
        
        int length1 = g_elements/2;
        int length2 = 0;
        //printf("length1 = %d\n", length1);
        
        float* result2;
        
        if (g_elements%2 == 0){
            result2 = calloc(g_elements/2, sizeof(float));
            result2 = memcpy(result2, matrix + g_elements/2, g_elements*sizeof(float)/2);
            length2 = g_elements/2;
        }
        else{
            result2 = calloc(g_elements/2 + 1, sizeof(float));
            result2 = memcpy(result2, matrix + g_elements/2, g_elements*sizeof(float)/2 + sizeof(float));
            length2 = g_elements/2 + 1;
        }
        
        sorted1_args* args1 = malloc(sizeof(sorted1_args));
        sorted2_args* args2 = malloc(sizeof(sorted2_args));

        //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
        args1[0] = (sorted1_args) {
            .id = 0,
            .result = result,
        };
        
        args2[0] = (sorted2_args){
            .id = 1,
            .result = result2,
        };
        
        pthread_t thread_ids[2];
        
        pthread_create(thread_ids, NULL, sorted1_worker, args1);
        pthread_create(thread_ids + 1, NULL, sorted2_worker, args2);
        
        pthread_join(thread_ids[0], NULL);
        pthread_join(thread_ids[1], NULL);
        
        /*printf("array 1: \n");
        for (int i = 0; i < length1; i++){
            printf(" %f ", args1[0].result[i]);
        }
        printf("\narray 2: \n");
        for (int i = 0; i < length2; i++){
            printf(" %f ", args2[0].result[i]);
        }*/

        //diff = clock() - start;
        //int msec = diff * 1000 / CLOCKS_PER_SEC;
        //printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
        int i = 0;
        int j = 0;
        int k = 0;
        
        float* toReturn = new_matrix();
        while (i < length1 && j < length2)
        {
            if (args1[0].result[i] < args2[0].result[j])
            {
                toReturn[k] = args1[0].result[i];
                //printf("just inputted %f into k = %d\n", toReturn[k], k);
                i++;
            }
            else
            {
                toReturn[k] = args2[0].result[j];
                //printf("just inputted %f into k = %d\n", toReturn[k], k);
                j++;
            }
            k++;
        }

        while (i < length1)
        {
            toReturn[k] = args1[0].result[i];
            //printf("just inputted %f into k = %d\n", toReturn[k], k);
            i++;
            k++;
        }

        while (j < length2)
        {
            toReturn[k] = args2[0].result[j];
            //printf("just inputted %f into k = %d\n", toReturn[k], k);
            j++;
            k++;
        }
        free(result);
        free(result2);
        free(args1);
        free(args2);
        
        //diff = clock() - start;
        //int msec = diff * 1000 / CLOCKS_PER_SEC;
        //printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
        
        return toReturn;
    }
    
    else{
        float* result = new_matrix();
        result = memcpy(result, matrix, g_elements*sizeof(float));
        qsort(result, g_width*g_height, sizeof(float), compareNum);
        
        //diff = clock() - start;
        //int msec = diff * 1000 / CLOCKS_PER_SEC;
        //printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
        return result;
    }
	
}

void* rotated_worker(void* args){
    rotated_args* wargs = (rotated_args*) args;
    
    int chunk = (g_width / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_width : (wargs->id + 1) * chunk;
    
    for (int j = start; j < end; j++){
        for (int i = 0; i < g_height; i++){
            wargs->result[i*g_width + j] = wargs->matrix[i + g_width*(g_width - 1 - j)];
        }
    }
    
    return NULL;
}

/**
 * Returns new matrix with elements rotated 90 degrees clockwise.
 */
float* rotated(const float* matrix) {

    float* result = new_matrix();
    
    rotated_args* args = malloc(sizeof(rotated_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (rotated_args) {
            .id = i,
            .result = result,
            .matrix = matrix,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, rotated_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    free(args);

	return result;
    
}

void* reversed_worker(void* args){
    
    //Copy the args into the sum_args struct
    reversed_args* wargs = (reversed_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    for (int i = start; i < end; i++){
        wargs->result[i] = wargs->matrix[g_elements-1-i];
    }
    
    return NULL;
}

/**
 * Returns new matrix with elements ordered in reverse.
 */
float* reversed(const float* matrix) {

	float* result = new_matrix();
    
    reversed_args* args = malloc(sizeof(reversed_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (reversed_args) {
            .id = i,
            .result = result,
            .matrix = matrix,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, reversed_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }

	free(args);

	return result;
}

void* transposed_worker(void* args){
    
    //Copy the args into the sum_args struct
    transposed_args* wargs = (transposed_args*) args;
    
    int chunk = (g_height / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_height : (wargs->id + 1) * chunk;
    
    for (ssize_t y = start; y < end; y++) {
		for (ssize_t x = 0; x < g_width; x++) {
			wargs->result[x * g_width + y] = wargs->matrix[y * g_width + x];
		}
	}
    
    return NULL;
}

/**
 * Returns new transposed matrix.
 */
float* transposed(const float* matrix) {

	float* result = new_matrix();
    
    transposed_args* args = malloc(sizeof(transposed_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (transposed_args) {
            .id = i,
            .result = result,
            .matrix = matrix,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, transposed_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    free(args);
	return result;
}

void* sAdd_worker(void* args){
    
    sAdd_args* wargs = (sAdd_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    for (size_t i = start; i < end; i++){
        wargs->result[i] = wargs->array[i] + wargs->scalar;
    }
    
    return NULL;
}

/**
 * Returns new matrix with scalar added to each element.
 */
float* scalar_add(const float* matrix, float scal) {

	float* res = new_matrix();
    
    sAdd_args* args = malloc(sizeof(sAdd_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (sAdd_args) {
            .id = i,
            .array = matrix,
            .result = res,
            .scalar = scal,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, sAdd_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    free(args);

	return res;
}

void* sMul_worker(void* args){
    
    sMul_args* wargs = (sMul_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    for (size_t i = start; i < end; i++){
        wargs->result[i] = wargs->array[i] * wargs->scalar;
    }
    
    return NULL;
}

/**
 * Returns new matrix with scalar multiplied to each element.
 */
float* scalar_mul(const float* matrix, float scal) {

	float* res = new_matrix();
    
    sMul_args* args = malloc(sizeof(sMul_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (sMul_args) {
            .id = i,
            .array = matrix,
            .result = res,
            .scalar = scal,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, sMul_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    free(args);

	return res;
}

void* matrixAdd_worker(void* args){
    
    matrixAdd_args* wargs = (matrixAdd_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    //printf("Thread %d is starting at %d and ending at %d\n", (int)wargs->id + 1, (int)start, (int)end);
    
    for (size_t i = start; i < end; i++){
        wargs->result[i] = wargs->matrix_a[i] + wargs->matrix_b[i];
    }
    
    return NULL;
}

/**
 * Returns new matrix that is the result of
 * adding the two given matrices together.
 */
float* matrix_add(const float* matrixa, const float* matrixb) {
    
    matrixAdd_args* args = malloc(sizeof(matrixAdd_args) * g_nthreads);
    float* res = new_matrix();
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (matrixAdd_args) {
            .id = i,
            .matrix_a = matrixa,
            .matrix_b = matrixb,
            .result = res,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, matrixAdd_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    free(args);
	return res;
}

void* matrixMul_worker(void* args){
    
    matrixMul_args* wargs = (matrixMul_args*) args;

    const size_t chunk = (g_width / g_nthreads);
    
	const size_t start = wargs->id * chunk;
	const size_t end = wargs->id == g_nthreads - 1 ? g_width : (wargs->id + 1) * chunk;

	for (size_t i = start; i < end; i++) {
        for (size_t k = 0; k < g_width; k++) {
		  for (size_t j = 0; j < g_width; j++) {
				wargs->result[((i) * g_width + (j))] += wargs->matrix_a[((i) * g_width + (k))] * wargs->matrix_b[((k) * g_width + (j))];
			}
		}
	}

	return NULL;
}

/**
 * Returns new matrix that is the result of
 * multiplying the two matrices together.
 */
float* matrix_mul(const float* a, const float* b) {

    float* result = calloc(g_width * g_width, sizeof(float));

	matrixMul_args args[g_nthreads];
	pthread_t thread_ids[g_nthreads];

	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (matrixMul_args) {
            .id = i,
			.matrix_a = a,
			.matrix_b = b,
			.result = result,
		};
	}

	// Launch threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, matrixMul_worker, args + i);
	}

	// Wait for threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	return result;
}

void* matrix_mulForPow_worker(void* args){
    
    matrixMul_args* wargs = (matrixMul_args*) args;

    const size_t chunk = (g_width / g_nthreads);
    
	const size_t start = wargs->id * chunk;
	const size_t end = wargs->id == g_nthreads - 1 ? g_width : (wargs->id + 1) * chunk;

	for (size_t i = start; i < end; i++) {
        for (size_t k = 0; k < g_width; k++) {
		  for (size_t j = 0; j < g_width; j++) {
				wargs->result[((i) * g_width + (j))] += wargs->matrix_a[((i) * g_width + (k))] * wargs->matrix_b[((k) * g_width + (j))];
			}
		}
	}

	return NULL;
    
}

float* matrix_mulForPow(const float* a, float* b) {

	float* result = calloc(g_width * g_width, sizeof(float));

	matrixMul_args args[g_nthreads];
	pthread_t thread_ids[g_nthreads];

	for (size_t i = 0; i < g_nthreads; i++) {
		args[i] = (matrixMul_args) {
            .id = i,
			.matrix_a = a,
			.matrix_b = b,
			.result = result,
		};
	}

	// Launch threads
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_create(thread_ids + i, NULL, matrixMul_worker, args + i);
	}

	// Wait for threads to finish
	for (size_t i = 0; i < g_nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}
    
    memcpy(b, result, g_elements*sizeof(float));
    free(result);

	return b;
}

/**
 * Returns new matrix that is the result of
 * powering the given matrix to the exponent.
 */
float* matrix_pow(const float* matrix, int exponent) {
    //clock_t start = clock(), diff;
	float* result = new_matrix();

    if (exponent == 0){
        for (ssize_t i = 0; i < g_width; i++) {
            result[i + i*g_width] = 1;
	   }
        return result;
    }
    else{
        //Clone the matrix
        for (ssize_t y = 0; y < g_height; y++) {
            for (ssize_t x = 0; x < g_width; x++) {
                result[y * g_width + x] = matrix[y * g_width + x];
            }
        }

        for (int i = 0; i < exponent - 1; i++){
            result = matrix_mulForPow(matrix, result);
        }
    }
    
    //diff = clock() - start;
    //int msec = diff * 1000 / CLOCKS_PER_SEC;
    //printf("Time taken %d seconds %d milliseconds", msec/1000, msec%1000);
    
	return result;
}

/**
 * Returns new matrix that is the result of
 * convolving given matrix with a 3x3 kernel matrix.
 */
float* matrix_conv(const float* matrix, const float* kernel) {
    
	float* result = new_matrix();

    //So that we're not writing to the same entry 9 times, instead only once
    float toInput = 0;
    
    //TOP ROW
    //Top left corner
    toInput += matrix[0] * kernel[0];
    toInput += matrix[0] * kernel[1];
    toInput += matrix[1] * kernel[2];
    toInput += matrix[0] * kernel[3];
    toInput += matrix[0] * kernel[4];
    toInput += matrix[1] * kernel[5];
    toInput += matrix[g_width] * kernel[6];
    toInput += matrix[g_width] * kernel[7];
    toInput += matrix[g_width + 1] * kernel[8];
    
    result[0] = toInput;
    toInput = 0;
    
    //Top edge
    for (int i = 1; i < (g_width - 1); i++){
        toInput += matrix[i - 1] * kernel[0];
        toInput += matrix[i] * kernel[1];
        toInput += matrix[i + 1] * kernel[2];
        toInput += matrix[i - 1] * kernel[3];
        toInput += matrix[i] * kernel[4];
        toInput += matrix[i + 1] * kernel[5];
        toInput += matrix[i + g_width - 1] * kernel[6];
        toInput += matrix[i + g_width ] * kernel[7];
        toInput += matrix[i + g_width + 1] * kernel[8];
        
        result[i] = toInput;
        toInput = 0;
    }
    
    //Top right corner
    toInput += matrix[g_width - 2] * kernel[0];
    toInput += matrix[g_width - 1] * kernel[1];
    toInput += matrix[g_width - 1] * kernel[2];
    toInput += matrix[g_width - 2] * kernel[3];
    toInput += matrix[g_width - 1] * kernel[4];
    toInput += matrix[g_width - 1] * kernel[5];
    toInput += matrix[2*g_width - 2] * kernel[6];
    toInput += matrix[2*g_width - 1] * kernel[7];
    toInput += matrix[2*g_width - 1] * kernel[8];
    
    result[g_width - 1] = toInput;
    toInput = 0;
    
    //MIDDLE ROWS
    for (int i = 1; i < g_height - 1; i++){
        
        //Left hand index
        toInput += matrix[i*g_height - g_height] * kernel[0];
        toInput += matrix[i*g_height - g_height] * kernel[1];
        toInput += matrix[i*g_height - g_height + 1] * kernel[2];
        toInput += matrix[i*g_height] * kernel[3];
        toInput += matrix[i*g_height] * kernel[4];
        toInput += matrix[i*g_height + 1] * kernel[5];
        toInput += matrix[i*g_height + g_width] * kernel[6];
        toInput += matrix[i*g_height + g_width] * kernel[7];
        toInput += matrix[i*g_height + g_width + 1] * kernel[8];
        
        result[i*g_height] = toInput;
        toInput = 0;
        
        //Middle indexes
        for (int j = 1; j < g_width - 1; j++){
            toInput += matrix[i*g_width + j - g_width - 1] * kernel[0];
            toInput += matrix[i*g_width + j - g_width] * kernel[1];
            toInput += matrix[i*g_width + j - g_width + 1] * kernel[2];
            toInput += matrix[i*g_width + j - 1] * kernel[3];
            toInput += matrix[i*g_width + j] * kernel[4];
            toInput += matrix[i*g_width + j + 1] * kernel[5];
            toInput += matrix[i*g_width + j + g_width - 1] * kernel[6];
            toInput += matrix[i*g_width + j + g_width] * kernel[7];
            toInput += matrix[i*g_width + j + g_width + 1] * kernel[8];
            
            result[i*g_width + j] = toInput;
            toInput = 0;
        }
        
        //Right hand index
        toInput += matrix[i*g_height - 2] * kernel[0];
        toInput += matrix[i*g_height - 1] * kernel[1];
        toInput += matrix[i*g_height - 1] * kernel[2];
        toInput += matrix[i*g_height + g_width - 2] * kernel[3];
        toInput += matrix[i*g_height + g_width - 1] * kernel[4];
        toInput += matrix[i*g_height + g_width - 1] * kernel[5];
        toInput += matrix[i*g_height + 2 * g_width - 2] * kernel[6];
        toInput += matrix[i*g_height + 2 * g_width - 1] * kernel[7];
        toInput += matrix[i*g_height + 2 * g_width - 1] * kernel[8];
        
        result[i*g_height + g_width - 1] = toInput;
        toInput = 0;
    }
    
    //BOTTOM ROW
    //Bottom left corner
    toInput += matrix[g_width*g_width - 2*g_width] * kernel[0];
    toInput += matrix[g_width*g_width - 2*g_width] * kernel[1];
    toInput += matrix[g_width*g_width - 2*g_width + 1] * kernel[2];
    toInput += matrix[g_width*g_width - g_width] * kernel[3];
    toInput += matrix[g_width*g_width - g_width] * kernel[4];
    toInput += matrix[g_width*g_width - g_width + 1] * kernel[5];
    toInput += matrix[g_width*g_width - g_width] * kernel[6];
    toInput += matrix[g_width*g_width - g_width] * kernel[7];
    toInput += matrix[g_width*g_width - g_width + 1] * kernel[8];
    
    result[g_width*g_width - g_width] = toInput;
    toInput = 0;
    
    //Bottom row
    for (int i = 1; i < g_width - 1; i++){
        toInput += matrix[g_width*g_width - g_width + i - g_width - 1] * kernel[0];
        toInput += matrix[g_width*g_width - g_width + i - g_width] * kernel[1];
        toInput += matrix[g_width*g_width - g_width + i - g_width + 1] * kernel[2];
        toInput += matrix[g_width*g_width - g_width + i - 1] * kernel[3];
        toInput += matrix[g_width*g_width - g_width + i] * kernel[4];
        toInput += matrix[g_width*g_width - g_width + i + 1] * kernel[5];
        toInput += matrix[g_width*g_width - g_width + i - 1] * kernel[6];
        toInput += matrix[g_width*g_width - g_width + i] * kernel[7];
        toInput += matrix[g_width*g_width - g_width + i + 1] * kernel[8];
        
        result[g_width*g_width - g_width + i] = toInput;
        toInput = 0;
    }
    
    //Bottom right corner
    toInput += matrix[g_width*g_width - g_width - 2] * kernel[0];
    toInput += matrix[g_width*g_width - g_width - 1] * kernel[1];
    toInput += matrix[g_width*g_width - g_width - 1] * kernel[2];
    toInput += matrix[g_width*g_width - 2] * kernel[3];
    toInput += matrix[g_width*g_width - 1] * kernel[4];
    toInput += matrix[g_width*g_width - 1] * kernel[5];
    toInput += matrix[g_width*g_width - 2] * kernel[6];
    toInput += matrix[g_width*g_width - 1] * kernel[7];
    toInput += matrix[g_width*g_width - 1] * kernel[8];
    
    result[g_width*g_width - 1] = toInput;
    
    return result;
}


////////////////////////////////
///       COMPUTATIONS       ///
////////////////////////////////

/**
 * Returns the sum of all elements.
 */

void* sum_worker(void* args){
    
    //Copy the args into the sum_args struct
    sum_args* wargs = (sum_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    //printf("Thread %d is starting at %d and ending at %d\n", (int)wargs->id + 1, (int)start, (int)end);
    
    double sum = 0;
    
    for (size_t i = start; i < end; i++){
        sum += wargs->array[i];
    }
    
    //printf("Thread %d computed sum to be %f\n", (int)wargs->id + 1, sum);
    
    wargs->result = sum;
    
    //printf("Thread %d added %f\n", (int)wargs->id + 1, wargs->result);
    
    return NULL;
    
}

float get_sum(const float* matrix) {
    
    sum_args* args = malloc(sizeof(sum_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (sum_args) {
            .id = i,
            .array = matrix,
            .result = 0,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, sum_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    double sum = 0;
    
    for (ssize_t i = 0; i < g_nthreads; i++) {
        sum += args[i].result;
    }
    
    //printf("Double sum %f\n", sum);
    
    free(args);
    
    
    
    return sum;
}

/**
 * Returns the trace of the matrix.
 */
float get_trace(const float* matrix) {
    
    float trace = 0;
    
    int i = 0;
    
    while (i < g_width){
        trace += (matrix[i*g_width + i]);
        i++;
    }

	return trace;
}

void* min_worker(void* args){
    
    min_args* wargs = (min_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    
    //printf("Thread %d is starting at %d and ending at %d\n", (int)wargs->id + 1, (int)start, (int)end);
    
    double min = wargs->array[0];
    
    for (size_t i = start; i < end; i++){
        if (wargs->array[i] < min){
            min = wargs->array[i];
        }
    }
    
    //printf("Thread %d computed sum to be %f\n", (int)wargs->id + 1, sum);
    
    wargs->result = min;
    
    //printf("Thread %d added %f\n", (int)wargs->id + 1, wargs->result);
    
    return NULL;
    
}

/**
 * Returns the smallest value in the matrix.
 */
float get_minimum(const float* matrix) {

	min_args* args = malloc(sizeof(min_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (min_args) {
            .id = i,
            .array = matrix,
            .result = 0,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, min_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    float min = args[0].result;
    
    for (ssize_t i = 1; i < g_nthreads; i++) {
        if (args[i].result < min){
            min = args[i].result;
        }
    }
    free(args);
	return min;
}

void* max_worker(void* args){
    max_args* wargs = (max_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    
    //printf("Thread %d is starting at %d and ending at %d\n", (int)wargs->id + 1, (int)start, (int)end);
    
    double max = wargs->array[0];
    
    for (size_t i = start; i < end; i++){
        if (wargs->array[i] > max){
            max = wargs->array[i];
        }
    }
    
    //printf("Thread %d computed sum to be %f\n", (int)wargs->id + 1, sum);
    
    wargs->result = max;
    
    //printf("Thread %d added %f\n", (int)wargs->id + 1, wargs->result);
    
    return NULL;
    
}

/**
 * Returns the largest value in the matrix.
 */
float get_maximum(const float* matrix) {
    
    max_args* args = malloc(sizeof(max_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (max_args) {
            .id = i,
            .array = matrix,
            .result = 0,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, max_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    float max = args[0].result;
    
    for (ssize_t i = 1; i < g_nthreads; i++) {
        if (args[i].result > max){
            max = args[i].result;
        }
    }
    free(args);
	return max;
}

float det(const float* matrix, float width){
    
    if (width == 2){
        float toReturn = (matrix[0]*matrix[3] - matrix[1]*matrix[2]);
        return toReturn;
    }
    
    else{
        float multiplier = 1;
        float determinant = 0;
        int rowToLookAt = 0;
        int j = 0;
        
        //Traverse through every element in the first row
        for (float col = 0; col < width; col++){
            rowToLookAt = col + width;
            
            //Make a new matrix which will be the same as the given one, just with the first row and current column taken out
            float* m = (float *) malloc((width*width - (width + width - 1))*sizeof(float));
            
            j = 0;
            
            //Make the matrix. It will take out the first row from the start
            for (int i = width; i < width*width; i++){
                
                //If we are at the column we are taking out, just skip through it
                if (i==rowToLookAt){
                    
                    rowToLookAt+=width;
                    continue;
                }
                
                else{
                    m[j] = matrix[i];
                    j++;
                }
            }
            determinant = (determinant + (matrix[(int)col])*(multiplier)*det(m, (width-1)));
            multiplier = multiplier * -1;
            free(m);
        }
        
        return determinant;
    }
}

/**
 * Returns the determinant of the matrix. This is the helper method to call det() recursively 
 */
float get_determinant(const float* matrix) {
    return det(matrix, g_width);
}

void* frequency_worker(void* args){
    
     //Copy the args into the sum_args struct
    freq_args* wargs = (freq_args*) args;
    
    int chunk = (g_elements / g_nthreads);
    
    const size_t start = wargs->id * chunk;
    
    //Check if we're at the final thread. If so, the end is the end of the given array. If not, we advance by the chunk size
    const size_t end = wargs-> id == g_nthreads - 1 ? g_elements : (wargs->id + 1) * chunk;
    
    //printf("Thread %d is starting at %d and ending at %d\n", (int)wargs->id + 1, (int)start, (int)end);
    
    double frequency = 0;
    
    for (size_t i = start; i < end; i++){
        if (wargs->array[i] == wargs->key){
            frequency++;
        }
    }
    
    //printf("Thread %d computed sum to be %f\n", (int)wargs->id + 1, sum);
    
    wargs->freq = frequency;
    
    //printf("Thread %d added %f\n", (int)wargs->id + 1, wargs->result);
    
    return NULL;
}

/**
 * Returns the frequency of the given value in the matrix.
 */
ssize_t get_frequency(const float* matrix, float value) {
    
    //clock_t start = clock(), diff;
    
    freq_args* args = malloc(sizeof(freq_args) * g_nthreads);
    
    //Make all the threads arguments. This is an array of structs, and we'll pick out values from this to pass to our threads when we make them
    for (size_t i = 0; i < g_nthreads; i++){
        args[i] = (freq_args) {
            .key = value,
            .id = i,
            .array = matrix,
            .freq = 0,
        };
    }
    
    pthread_t thread_ids[g_nthreads];
    
    //Launch threads
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_create(thread_ids + i, NULL, frequency_worker, args + i);
    }
    
    //Wait for threads to finish
    for (size_t i = 0; i < g_nthreads; i++){
        pthread_join(thread_ids[i], NULL);
    }
    
    ssize_t frequency = 0;
    
    for (ssize_t i = 0; i < g_nthreads; i++){
        frequency += args[i].freq;
    }
    
    //diff = clock() - start;
    //int msec = diff * 1000 / CLOCKS_PER_SEC;
    //printf("Time taken %d seconds %d milliseconds", msec/1000, msec%1000);
    
    free(args);

	return frequency;
}
