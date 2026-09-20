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
// weight out; see ptq1_0_trit below for how one is read off a byte.
#define QK_PTQ1_0 128
typedef struct {
    uchar qs[24];  // 5 trits per byte: 16 strided bytes -> 80 weights, 8 -> 40
    uchar qh[2];   // 4 trits per byte -> 8 weights
    half  d;
} block_ptq1_0;

// Weights per thread per step. Four keeps a thread inside one block and one
// base-3 stage, so a step needs a single scale and a single pow3.
#define NB_PTQ1_0 4

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

inline float ptq1_0_trit(uchar byte, uchar p) {
    const uchar q = (uchar) (byte * p);

    return (float) (((uint) q * 3) >> 8) - 1.0f;  // 0, 1, 2 -> -1, 0, 1
}

// A block is 28 bytes and the byte offset of a step is a multiple of four, so
// the four bytes are consecutive and 4-aligned: one vector load, not four.
#define PTQ1_0_LOAD4(qs, m) vload4((m) >> 2, (global const uchar *) (qs))

// r is the weight index inside the block, a multiple of NB_PTQ1_0.
inline float ptq1_0_dot4(global const block_ptq1_0 * qb, int r,
                         float y0, float y1, float y2, float y3) {
    if (r < 80) {
        const uchar4 v = PTQ1_0_LOAD4(qb->qs, r & 15);
        const uchar  p = ptq1_0_pow3(r >> 4);

        return y0 * ptq1_0_trit(v.x, p) + y1 * ptq1_0_trit(v.y, p)
             + y2 * ptq1_0_trit(v.z, p) + y3 * ptq1_0_trit(v.w, p);
    }
    if (r < 120) {
        const int s = r - 80;
        const uchar4 v = PTQ1_0_LOAD4(qb->qs, 16 + (s & 7));
        const uchar  p = ptq1_0_pow3(s >> 3);

        return y0 * ptq1_0_trit(v.x, p) + y1 * ptq1_0_trit(v.y, p)
             + y2 * ptq1_0_trit(v.z, p) + y3 * ptq1_0_trit(v.w, p);
    }

    const int    s = r - 120;
    const uchar2 h = vload2(0, (global const uchar *) qb->qh);

    return y0 * ptq1_0_trit((s & 1) ? h.y : h.x,             ptq1_0_pow3((s + 0) >> 1))
         + y1 * ptq1_0_trit(((s + 1) & 1) ? h.y : h.x,       ptq1_0_pow3((s + 1) >> 1))
         + y2 * ptq1_0_trit(((s + 2) & 1) ? h.y : h.x,       ptq1_0_pow3((s + 2) >> 1))
         + y3 * ptq1_0_trit(((s + 3) & 1) ? h.y : h.x,       ptq1_0_pow3((s + 3) >> 1));
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

    // A block's 128 weights are spread over three strided runs of the base-3
    // layout. Walking them one run at a time keeps every lane in the same branch
    // of ptq1_0_dot4, where a mixed walk would make the subgroup serialize all
    // three runs on every step.
    #define PTQ1_0_STAGE_LOOP(NSLOT, R_OF_SLOT)                                       \
    for (int i = lane; i < nb*NSLOT; i += N_SIMDWIDTH) {                               \
        const int blk = i / NSLOT;                                                     \
        const int r   = (R_OF_SLOT);                                                   \
        const int e   = blk*QK_PTQ1_0 + r;                                             \
                                                                                       \
        const float y0 = y[e + 0];                                                     \
        const float y1 = y[e + 1];                                                     \
        const float y2 = y[e + 2];                                                     \
        const float y3 = y[e + 3];                                                     \
                                                                                       \
        for (int row = 0; row < N_R0_PTQ1_0; ++row) {                                  \
            global const block_ptq1_0 * qb = ax[row] + blk;                            \
                                                                                       \
            sumf[row] += (float) qb->d * ptq1_0_dot4(qb, r, y0, y1, y2, y3);           \
        }                                                                              \
    }

    PTQ1_0_STAGE_LOOP(20, (i - blk*20)*4)          // weights   0..79  in qs[0..15]
    PTQ1_0_STAGE_LOOP(10, 80 + (i - blk*10)*4)     // weights  80..119 in qs[16..23]
    PTQ1_0_STAGE_LOOP(2,  120 + (i - blk*2)*4)     // weights 120..127 in qh[0..1]

    #undef PTQ1_0_STAGE_LOOP

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    for (int row = 0; row < N_R0_PTQ1_0; ++row) {
        float tot = sub_group_reduce_add(sumf[row]);

        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}
