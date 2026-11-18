/* Decoder of the run-length encoding of .goo layers, for nemnest.
 * A plain C function loaded with ctypes: no Python headers needed.
 * Returns the number of runs written, or -1 if the data is truncated. */
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int64_t nemnest_goo_runs(const uint8_t *data, int64_t n, int64_t *lengths, uint8_t *values)
{
    int64_t k = 0, count = 0;
    unsigned color = 0;
    while (k < n) {
        unsigned head = data[k];
        unsigned kind = head >> 6, size = (head >> 4) & 3;
        int64_t length;
        if (kind == 2) {                      /* colour = previous +/- a small difference */
            unsigned diff = head & 0x0F;
            if (size & 1) {
                if (k + 1 >= n) return -1;
                k++;
                length = data[k];
            } else {
                length = 1;
            }
            color = ((size & 2) ? color - diff : color + diff) & 0xFF;
        } else {
            if (kind == 0) {
                color = 0;
            } else if (kind == 3) {
                color = 255;
            } else {                          /* explicit grey value */
                if (k + 1 >= n) return -1;
                k++;
                color = data[k];
            }
            length = head & 0x0F;
            if (k + (int64_t)size >= n && size) return -1;
            if (size == 1) {
                length |= (int64_t)data[k + 1] << 4;
            } else if (size == 2) {
                length |= ((int64_t)data[k + 1] << 12) | ((int64_t)data[k + 2] << 4);
            } else if (size == 3) {
                length |= ((int64_t)data[k + 1] << 20) | ((int64_t)data[k + 2] << 12) | ((int64_t)data[k + 3] << 4);
            }
            k += size;
        }
        k++;
        lengths[count] = length;
        values[count] = (uint8_t)color;
        count++;
    }
    return count;
}

/* setuptools builds this file as an extension module and, on Windows, asks the
 * linker to export this symbol. The library is only ever loaded with ctypes. */
EXPORT void *PyInit__runs(void) { return 0; }
