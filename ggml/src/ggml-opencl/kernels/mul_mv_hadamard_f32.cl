#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define HADAMARD_MAX_N 2048
#define FWHT_NTH       256

typedef struct { float v[HADAMARD_MAX_N]; } fwht_buf_t;

// dst = H * src1 for the normalized Sylvester-Walsh matrix H (entries
// (-1)^popcount(i&j) / sqrt(n)), computed as a butterfly transform so that the
// explicit n x n matrix never has to be read.
kernel void kernel_mul_mat_hadamard_f32(
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        int ncols,
        int n,
        ulong nb11,
        ulong nbd1,
        float scale) {

    global const float * restrict x = (global const float *)(src1 + offset1);
    global float * restrict y =       (global float *)     (dst  + offsetd);

    const int group = get_group_id(0);
    const int blk   = group % ncols;
    const int col   = group / ncols;
    const int tid   = get_local_id(0);
    const int nth   = get_local_size(0);

    const ulong s11 = nb11 / 4;
    const ulong sd1 = nbd1 / 4;
    const long base_x = (long) col * s11 + (long) blk * n;
    const long base_y = (long) col * sd1 + (long) blk * n;

    __local fwht_buf_t buf;

    for (int i = tid; i < n; i += nth) {
        buf.v[i] = x[base_x + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int h = 1; h < n; h <<= 1) {
        for (int i = tid; i < (n >> 1); i += nth) {
            const int j = ((i & ~(h - 1)) << 1) | (i & (h - 1));
            const float a = buf.v[j];
            const float b = buf.v[j + h];
            buf.v[j]     = a + b;
            buf.v[j + h] = a - b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = tid; i < n; i += nth) {
        y[base_y + i] = buf.v[i] * scale;
    }
}
