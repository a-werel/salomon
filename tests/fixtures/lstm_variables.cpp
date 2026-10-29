int original(int a, int b) {
    int result = a;
    result = b;
    return result;
}

int renamed(int left, int right) {
    int output = left;
    output = right;
    return output;
}

int different(int a, int b) {
    int result = a;
    result = a;
    return result;
}

int shadowed(int value) {
    int result = value;
    {
        int value = 1;
        result = value;
    }
    return value;
}
