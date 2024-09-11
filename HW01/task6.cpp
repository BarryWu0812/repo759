#include <iostream>

int main(int argc, char* argv[])
{
    std::cout << "There are " << argc << " arguments:\n";
    int N = std::atoi(argv[1]);

    // Loop through each argument and print its number and value
    for (int count = 0; count <= N; count++)
    {
        if(count != N)
            printf("%d ", count);
        else
            printf("%d", count);
    }
    printf("\n");
    for (int count = N; count >= 0; count--)
    {
        if(count != 0)
            std::cout << count << ' ';
        else
            std::cout << count;
    }
    std::cout << '\n';
    return 0;
}