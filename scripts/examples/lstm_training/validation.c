int read_unchecked(const int *values, int index, int count) {
    return values[index];
}

int read_checked(const int *values, int index, int count) {
    if (index < 0 || index >= count) {
        return 0;
    }
    return values[index];
}
