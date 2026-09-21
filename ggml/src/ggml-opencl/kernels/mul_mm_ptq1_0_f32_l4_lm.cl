#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tiled GEMM for PTQ1_0, mirroring mul_mm_q2_0_f32_l4_lm: a workgroup owns a
// BM x BN tile of the output, stages one BK slice of dequantized src0 and of
// src1 through local memory, and reads each weight once per BK slice instead
// of once per column.
//
// Unlike q2_0 there is no flattened form to read - 128 trits at 2 bits need 32
// bytes and the block has 26 after the scale - so the loader addresses the
// stored block directly. LOAD_VEC_A stays 4 because inside one trit run
// consecutive weights are consecutive bytes.
#define LOAD_VEC_A 4
#define LOAD_VEC_B 4

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 8

#define QK_PTQ1_0 128
#define BYTES_PTQ1_0 28

// The 128 weights of a block are stored as three strided runs: 0..79 in
// qs[0..15] (5 trits per byte), 80..119 in qs[16..23] and 120..127 in qh.
// dequantize_row_ptq1_0 packs five trits of a run in one byte as
// ceil(q * 256 / 243), so multiplying by 3^n wraps mod 256 and leaves the n-th
// trit in the top bits.
inline float ptq1_0_weight(global const uchar * b, int k) {
    int bi;
    int pj;

    if (k < 80) {
        bi = k & 15;
        pj = k >> 4;
    } else if (k < 120) {
        const int s = k - 80;
        bi = 16 + (s & 7);
        pj = s >> 3;
    } else {
        const int s = k - 120;
        bi = 24 + (s & 1);
        pj = s >> 1;
    }

    const uchar p = pj == 0 ? 1 : pj == 1 ? 3 : pj == 2 ? 9 : pj == 3 ? 27 : 81;
    const uchar q = (uchar) (b[bi] * p);
    const float t = (float) (((uint) q * 3) >> 8) - 1.0f;

    return t * *(global const half *)(b + 26);
}

kernel void kernel_mul_mm_ptq1_0_f32_l4_lm(
    global uchar  * src0,
    ulong offset0,
    global float4 * src1,
    ulong offset1,
    global float  * dst,
    ulong offsetd,

    int ne00,
    int ne01,
    int ne02,
    int ne11,
    int ne12,

    int stride_a,
    int stride_b,
    int stride_d,

    int batch_stride_a,
    int batch_stride_b,
    int batch_stride_d,

    int r2,
    int r3
) {
    src0 = (global uchar *)((global char *)src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst  = (global float *) ((global char*)dst  + offsetd);

    local float buf_a[BM * BK];
    local float buf_b[BN * BK];

    const int batch_idx = get_global_id(2);

    const int i13 = batch_idx / ne12;
    const int i12 = batch_idx % ne12;

    const int i03 = i13 / r3;
    const int i02 = i12 / r2;

    const int batch_idx_a = i03 * ne02 + i02;

    const int ir = get_group_id(0);
    const int ic = get_group_id(1);

    const int tid = get_local_id(0);
    const int th_r  = tid % (BM / TM);
    const int th_c  = tid / (BM / TM);

    const int loadr_a = get_local_id(0) % (BK / LOAD_VEC_A);
    const int loadc_a = get_local_id(0) / (BK / LOAD_VEC_A);
    const int loadr_b = get_local_id(0) % (BK / LOAD_VEC_B);
    const int loadc_b = get_local_id(0) / (BK / LOAD_VEC_B);

    const int loadstride_a = get_local_size(0) * LOAD_VEC_A / BK;
    const int loadstride_b = get_local_size(0) * LOAD_VEC_B / BK;

    const int pos_a = batch_idx_a * batch_stride_a + ir * BM * stride_a;
    int pos_b = (batch_idx * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;

    float sums[TM * TN];
    float cache_a[TM];
    float cache_b[TN];

    for (int i = 0; i < TM * TN; i++) {
        sums[i] = 0.0f;
    }

    for (int block = 0; block < ne00; block += BK) {
        for (int l = 0; l < BM; l += loadstride_a) {
            const int row = ir*BM + loadc_a + l;

            if (row < ne01) {
                // rows are contiguous in weight units, so the block holding
                // weight k of row r is (r*ne00 + k) / QK_PTQ1_0
                const int k = block + loadr_a * LOAD_VEC_A;
                const int base = pos_a + (loadc_a + l) * stride_a;
                global const uchar * b = src0 +
                    (long)(base + k) / QK_PTQ1_0 * BYTES_PTQ1_0;
                const int kb = k % QK_PTQ1_0;

                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = ptq1_0_weight(b, kb + 0);
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = ptq1_0_weight(b, kb + 1);
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = ptq1_0_weight(b, kb + 2);
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = ptq1_0_weight(b, kb + 3);
            } else {
                for (int b = 0; b < LOAD_VEC_A; ++b) {
                    buf_a[(loadr_a * LOAD_VEC_A + b) * BM + loadc_a + l] = 0.0f;
                }
            }
        }

        for (int l = 0; l < BN; l += loadstride_b) {
            if (ic*BN + loadc_b + l < ne11) {
                int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = src1[idx].s0;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = src1[idx].s1;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = src1[idx].s2;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = src1[idx].s3;
            } else {
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        pos_b += BK / LOAD_VEC_B;

        for (int i = 0; i < BK; i++) {
            for (int j = 0; j < TM; j++) {
                cache_a[j] = buf_a[(i) * BM + th_r * TM + j];
            }

            for (int j = 0; j < TN; j++) {
                cache_b[j] = buf_b[(i) * BN + th_c * TN + j];
            }

            for (int cc = 0; cc < TN; cc++) {
                for (int cr = 0; cr < TM; cr++) {
                    const int sums_idx = cc*TM + cr;
                    sums[sums_idx] = mad(cache_a[cr], cache_b[cc], sums[sums_idx]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int dr = ir * BM + th_r * TM;
    const int dc = ic * BN + th_c * TN;

    const int offsets = batch_idx * batch_stride_d;

    for (int cc = 0; cc < TN; cc++) {
        for (int cr = 0; cr < TM; cr++) {
            if (dr + cr < ne01 && dc + cc < ne11) {
                dst[offsets + (dc + cc) * stride_d + dr + cr] = sums[cc * TM + cr];
            }
        }
    }
}
