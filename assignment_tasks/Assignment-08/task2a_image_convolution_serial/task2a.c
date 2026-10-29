#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define KSIZE 3

void convolution(int *input, int *output, int n)
{
    int kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

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

                    if (x >= 0 && x < n && y >= 0 && y < n)
                    {
                        sum += input[x * n + y] * kernel[u + 1][v + 1];
                    }
                }
            }

            output[i * n + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    printf("Matrix Size: %d x %d\n", n, n);

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

    clock_t start = clock();

    convolution(input, output, n);

    clock_t end = clock();

    double time_taken =
        (double)(end - start) / CLOCKS_PER_SEC;

    printf("Sequential Execution Time: %.6f seconds\n",
           time_taken);

    printf("Output[0][0] = %d\n", output[0]);
    printf("Output[%d][%d] = %d\n",
           n - 1, n - 1, output[n * n - 1]);

    free(input);
    free(output);

    return 0;
}