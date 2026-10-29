#include <stddef.h>
#include <string.h>

void copy_unchecked(char *destination, const char *source) {
    strcpy(destination, source);
}

int copy_checked(char *destination, size_t capacity, const char *source) {
    size_t length = strlen(source);
    if (length >= capacity) {
        return 0;
    }
    memcpy(destination, source, length + 1);
    return 1;
}
