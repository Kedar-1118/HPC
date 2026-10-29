#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void convolution(int *input, int *output, int n,
                 int schedule_type)
{
    int kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    if (schedule_type == 0)
    {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;

                for (int u = -1; u <= 1; u++)
                {
                    for (int v = -1; v <= 1; v++)
                    {
                        int x = i + u;
                        int y = j + v;

                        if (x >= 0 && x < n &&
                            y >= 0 && y < n)
                        {
                            sum += input[x * n + y]
                                   * kernel[u + 1][v + 1];
                        }
                    }
                }

                output[i * n + j] = sum;
            }
        }
    }
    else if (schedule_type == 1)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;

                for (int u = -1; u <= 1; u++)
                {
                    for (int v = -1; v <= 1; v++)
                    {
                        int x = i + u;
                        int y = j + v;

                        if (x >= 0 && x < n &&
                            y >= 0 && y < n)
                        {
                            sum += input[x * n + y]
                                   * kernel[u + 1][v + 1];
                        }
                    }
                }

                output[i * n + j] = sum;
            }
        }
    }
    else
    {
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;

                for (int u = -1; u <= 1; u++)
                {
                    for (int v = -1; v <= 1; v++)
                    {
                        int x = i + u;
                        int y = j + v;

                        if (x >= 0 && x < n &&
                            y >= 0 && y < n)
                        {
                            sum += input[x * n + y]
                                   * kernel[u + 1][v + 1];
                        }
                    }
                }

                output[i * n + j] = sum;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <matrix_size> <threads> <schedule>\n", argv[0]);
        printf("schedule: 0=static, 1=dynamic, 2=guided\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    int schedule_type = atoi(argv[3]);

    omp_set_num_threads(threads);

    int *input = malloc((size_t)n * n * sizeof(int));
    int *output = malloc((size_t)n * n * sizeof(int));

    if (input == NULL || output == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            input[i * n + j] = 1;
        }
    }

    double start = omp_get_wtime();

    convolution(input, output, n, schedule_type);

    double end = omp_get_wtime();

    const char *schedule_name;

    if (schedule_type == 0)
        schedule_name = "static";
    else if (schedule_type == 1)
        schedule_name = "dynamic";
    else
        schedule_name = "guided";

    printf("Matrix Size: %d x %d\n", n, n);
    printf("Threads: %d\n", threads);
    printf("Schedule: %s\n", schedule_name);
    printf("OpenMP Execution Time: %.6f seconds\n",
           end - start);

    printf("C[0][0] = %d\n", output[0]);

    free(input);
    free(output);

    return 0;
}