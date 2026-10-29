#include <stdio.h>

void print_unchecked(const char *message) {
    printf(message);
}

void print_checked(const char *message) {
    printf("%s", message);
}
