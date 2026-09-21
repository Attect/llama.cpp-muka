#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// PTQ1_0: 128 ternary weights and one fp16 scale per block, 1.75 bpw. Unlike
// Q1_0/Q2_0 the quants are a base-3 code, so there is no bit mask to pull a
// weight out; see ptq1_0_q below for how one is read off a byte.
#define QK_PTQ1_0 128
typedef struct {
    uchar qs[24];  // 5 trits per byte: 16 strided bytes -> 80 weights, 8 -> 40
    uchar qh[2];   // 4 trits per byte -> 8 weights
    half  d;
} block_ptq1_0;

#ifdef INTEL_GPU
#define N_R0_PTQ1_0 4 // number of rows each subgroup works on
#define N_SG_PTQ1_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_PTQ1_0 4
#define N_SG_PTQ1_0 2
#define N_SIMDWIDTH 64
#endif

// dequantize_row_ptq1_0 stores five trits of a strided run in one byte as
// ceil(q * 256 / 243), q being the base-3 number with the run's first weight in
// the most significant trit. Multiplying by 3^n wraps mod 256 so that the n-th
// trit is left in the top bits, and three bits of it are enough to hold 2*3.
inline uchar ptq1_0_pow3(int n) {
    return n == 0 ? 1 : n == 1 ? 3 : n == 2 ? 9 : n == 3 ? 27 : 81;
}

// The trit as 0/1/2. The weight is (t - 1) * d, and the -1 comes out of the
// inner product: sum((t-1)*y) == sum(t*y) - sum(y), where sum(y) is shared by
// all rows.
inline float ptq1_0_q(uchar byte, uchar p) {
    const uchar q = (uchar) (byte * p);

    return (float) (((uint) q * 3) >> 8);
}

// Consecutive weights inside one trit run are consecutive bytes. Blocks are 28
// bytes apart, so a quant load can be four-aligned but never eight-aligned -
// hence float4 here.
inline float4 ptq1_0_q4(global const uchar * q, uchar p) {
    const uchar4 v = vload4(0, q);

    return (float4)(ptq1_0_q(v.x, p), ptq1_0_q(v.y, p), ptq1_0_q(v.z, p), ptq1_0_q(v.w, p));
}

inline float ptq1_0_sum4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_ptq1_0_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0*N_SG_PTQ1_0 + get_sub_group_id()) * N_R0_PTQ1_0;

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    ulong offset_src1 = r1*nb11 + (ulong)i12*nb12 + (ulong)i13*nb13;
    global float * y  = (global float *) (src1 + offset_src1);

    // pointers to src0 rows
    global const block_ptq1_0 * ax[N_R0_PTQ1_0];
    for (int row = 0; row < N_R0_PTQ1_0; ++row) {
        ulong offset_src0 = (ulong)(first_row + row)*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
        ax[row] = (global const block_ptq1_0 *) ((global const char *) src0 + offset_src0);
    }

    float sumf[N_R0_PTQ1_0] = { 0.f };

    const int lane = get_sub_group_local_id();
    const int nb   = ne00/QK_PTQ1_0;

    // A block's 128 weights sit in three strided runs of the base-3 layout:
    // 0..79 in qs[0..15] (5 runs of 16), 80..119 in qs[16..23] (5 runs of 8) and
    // 120..127 in qh. Inside a run consecutive weights are consecutive bytes
    // under one pow3, so a lane takes a whole run at a time: its y values are
    // contiguous, and all four rows read the same byte offsets.
    #define PTQ1_0_ROWS(ACC_EXPR)                                                        \
        for (int row = 0; row < N_R0_PTQ1_0; ++row) {                                    \
            global const block_ptq1_0 * qb = ax[row] + blk;                              \
            sumf[row] += (float) qb->d * (ACC_EXPR);                                     \
        }

    for (int i = lane; i < nb*5; i += N_SIMDWIDTH) {
        const int blk = i / 5;
        const int j   = i - blk*5;

        global const float * yv = y + blk*QK_PTQ1_0 + j*16;
        const float4 y0 = vload4(0, yv);
        const float4 y1 = vload4(1, yv);
        const float4 y2 = vload4(2, yv);
        const float4 y3 = vload4(3, yv);
        const float  sy = ptq1_0_sum4(y0) + ptq1_0_sum4(y1) + ptq1_0_sum4(y2) + ptq1_0_sum4(y3);
        const uchar  p  = ptq1_0_pow3(j);

        PTQ1_0_ROWS(dot(y0, ptq1_0_q4(qb->qs,      p)) + dot(y1, ptq1_0_q4(qb->qs + 4,  p))
                  + dot(y2, ptq1_0_q4(qb->qs + 8,  p)) + dot(y3, ptq1_0_q4(qb->qs + 12, p)) - sy)
    }

    for (int i = lane; i < nb*5; i += N_SIMDWIDTH) {
        const int blk = i / 5;
        const int j   = i - blk*5;

        global const float * yv = y + blk*QK_PTQ1_0 + 80 + j*8;
        const float4 y0 = vload4(0, yv);
        const float4 y1 = vload4(1, yv);
        const float  sy = ptq1_0_sum4(y0) + ptq1_0_sum4(y1);
        const uchar  p  = ptq1_0_pow3(j);

        PTQ1_0_ROWS(dot(y0, ptq1_0_q4(qb->qs + 16, p)) + dot(y1, ptq1_0_q4(qb->qs + 20, p)) - sy)
    }

    // qh holds the last 8 weights interleaved: even ones in qh[0], odd in qh[1]
    for (int i = lane; i < nb; i += N_SIMDWIDTH) {
        const int blk = i;

        global const float * yv = y + blk*QK_PTQ1_0 + 120;
        const float4 y0 = vload4(0, yv);
        const float4 y1 = vload4(1, yv);
        const float  sy = ptq1_0_sum4(y0) + ptq1_0_sum4(y1);

        PTQ1_0_ROWS(dot(y0, (float4)(ptq1_0_q(qb->qh[0], 1),  ptq1_0_q(qb->qh[1], 1),
                                     ptq1_0_q(qb->qh[0], 3),  ptq1_0_q(qb->qh[1], 3)))
                  + dot(y1, (float4)(ptq1_0_q(qb->qh[0], 9),  ptq1_0_q(qb->qh[1], 9),
                                     ptq1_0_q(qb->qh[0], 27), ptq1_0_q(qb->qh[1], 27))) - sy)
    }

    #undef PTQ1_0_ROWS

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    for (int row = 0; row < N_R0_PTQ1_0; ++row) {
        float tot = sub_group_reduce_add(sumf[row]);

        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}
