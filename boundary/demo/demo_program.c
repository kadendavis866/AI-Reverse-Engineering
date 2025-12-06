#include <stdio.h>
#include <string.h>
#include <stdint.h>

static int add(int a, int b) {
    return a + b;
}

static int mul(int a, int b) {
    return a * b;
}

static int checksum(const uint8_t *buf, size_t len) {
    int sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += buf[i];
    }
    return sum & 0xFFFF;
}

static void greet(const char *name) {
    printf("Hello, %s!\n", name);
}

static void scramble(char *buf) {
    size_t len = strlen(buf);
    for (size_t i = 0; i + 1 < len; i += 2) {
        char tmp = buf[i];
        buf[i] = buf[i + 1];
        buf[i + 1] = tmp;
    }
}

int main(int argc, char **argv) {
    int a = 7;
    int b = 3;
    int s = add(a, b);
    int p = mul(a, b);
    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int cs = checksum(data, sizeof(data));

    char name[32] = "AI4CYBER";
    greet(name);
    scramble(name);
    printf("a+b=%d a*b=%d checksum=%d scrambled=%s\n", s, p, cs, name);
    return 0;
}
