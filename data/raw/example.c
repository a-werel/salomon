int calculate(int x) {
    int y = x + 1;
    if (y > 5) {
        y = y * 2;
    }
    return y;
}

int main() {
    return calculate(4);
}