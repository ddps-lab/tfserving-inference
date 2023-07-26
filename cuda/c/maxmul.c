#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#define SIZE 600 // 행렬의 크기
#define NUM_THREADS 1000 // 사용할 스레드 수
#define REQUEST_TIME 10

void maxmul(int *A, int* B, int *C, int size);

// 쓰레드 인자로 쓸 구조체 정의
struct thread_args {
    int thread_id;
    struct timeval st;
};

// 입력 행렬 A와 B
int A[SIZE*SIZE];
int B[SIZE*SIZE];

// 스레드 함수
void *multiply(void *arg) {
    struct thread_args* args = (struct thread_args*)arg;
    int thread_id = args->thread_id;
    struct timeval st = args->st;

    // 고유한 행렬 생성
    int matrix_A[SIZE*SIZE];
    int matrix_B[SIZE*SIZE];
    int matrix_C[SIZE*SIZE];

    // 행렬 A와 B 복사
    for (int i = 0; i < SIZE*SIZE; i++) {
        matrix_A[i] = A[i];
        matrix_B[i] = B[i];
    }

    // 행렬 곱셈 수행
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL); // 시작 시간 기록
    maxmul(matrix_A, matrix_B, matrix_C, SIZE);
    gettimeofday(&end_time, NULL); // 종료 시간 기록

    // 처리 시간 출력
    double execution_time = (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec) / 1000000;
    printf("Thread %d, matrix_result[0]=%d, start time: %.4f, end time: %.4f, processing time: %.4f seconds\n", thread_id, matrix_C[0], ((double)start_time.tv_sec + (double)start_time.tv_usec / 1000000) - ((double)st.tv_sec + (double)st.tv_usec / 1000000), ((double)end_time.tv_sec + (double)end_time.tv_usec / 1000000) - ((double)st.tv_sec + (double)st.tv_usec / 1000000), execution_time);

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    struct thread_args thread_args[NUM_THREADS];
    int num_thread_per_sec = NUM_THREADS / REQUEST_TIME;
    float usleep_time = ((float)1 / (float)num_thread_per_sec) * 1000000;

    // 행렬 A와 B를 초기화
    for (int i = 0; i < SIZE*SIZE; i++) {
        A[i] = rand()%100;
        B[i] = rand()%100;
    }

    struct timeval start_time;
    gettimeofday(&start_time, NULL); // 시작 시간 기록

    // 스레드 생성 및 행렬 곱셈 수행
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].st = start_time;
        pthread_create(&threads[i], NULL, multiply, (void *)&thread_args[i]);

        usleep((int)usleep_time);
    }

    // 모든 스레드의 종료를 기다림
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}