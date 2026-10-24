//go:build !purego

#include "textflag.h"

// Int8 tile kernels for processors with VNNI. Each computes
//
//	C[6][16] += sum over blocks b of da[r][b]·dw[c][b]·(Σ ua·w − 128·Σ w)
//
// where ua = qa+128 are unsigned activation bytes and w the signed Q8_0
// weights of one packed panel (see packQ8Panels). The two versions differ
// only in how vpdpbusd is encoded: EVEX (AVX512-VNNI with AVX512-VL) or
// VEX (AVX-VNNI, Intel 12th generation onward). The VEX bytes are written
// out because the Go assembler only knows the EVEX form.
//
// Registers: Y0, Y1 a panel row pair of weights; Y2, Y3 a broadcast group
// of four activation bytes; Y4-Y15 the int32 sums, two per row.

// DPn is vpdpbusd Yn, Y(2 or 3), Y(0 or 1).
#define DP4 VPDPBUSD Y0, Y2, Y4
#define DP5 VPDPBUSD Y1, Y2, Y5
#define DP6 VPDPBUSD Y0, Y3, Y6
#define DP7 VPDPBUSD Y1, Y3, Y7
#define DP8 VPDPBUSD Y0, Y2, Y8
#define DP9 VPDPBUSD Y1, Y2, Y9
#define DP10 VPDPBUSD Y0, Y3, Y10
#define DP11 VPDPBUSD Y1, Y3, Y11
#define DP12 VPDPBUSD Y0, Y2, Y12
#define DP13 VPDPBUSD Y1, Y2, Y13
#define DP14 VPDPBUSD Y0, Y3, Y14
#define DP15 VPDPBUSD Y1, Y3, Y15

// STEP runs four of a block's 32 values: activation byte offset oa,
// weight offsets o0 and o0+32.
#define STEP(oa, o0, o1) \
	VMOVDQU o0(DX), Y0 \
	VMOVDQU o1(DX), Y1 \
	VPBROADCASTD oa(SI), Y2 \
	DP4 \
	DP5 \
	VPBROADCASTD oa(SI)(R8*1), Y3 \
	DP6 \
	DP7 \
	VPBROADCASTD oa(SI)(R8*2), Y2 \
	DP8 \
	DP9 \
	VPBROADCASTD oa(R10), Y3 \
	DP10 \
	DP11 \
	VPBROADCASTD oa(R10)(R8*1), Y2 \
	DP12 \
	DP13 \
	VPBROADCASTD oa(R10)(R8*2), Y3 \
	DP14 \
	DP15

// FIN folds one row's block sums into C: subtract 128·Σw, scale by the
// weight scales and the row's activation scale, add to C.
#define FIN(acc0, acc1, sc, c0, c1) \
	VPSUBD 576(DX), acc0, acc0 \
	VPSUBD 608(DX), acc1, acc1 \
	VCVTDQ2PS acc0, acc0 \
	VCVTDQ2PS acc1, acc1 \
	VMULPS 512(DX), acc0, acc0 \
	VMULPS 544(DX), acc1, acc1 \
	VBROADCASTSS sc, Y2 \
	VFMADD213PS c0, Y2, acc0 \
	VFMADD213PS c1, Y2, acc1 \
	VMOVUPS acc0, c0 \
	VMOVUPS acc1, c1

#define ZERO \
	VPXOR Y4, Y4, Y4 \
	VPXOR Y5, Y5, Y5 \
	VPXOR Y6, Y6, Y6 \
	VPXOR Y7, Y7, Y7 \
	VPXOR Y8, Y8, Y8 \
	VPXOR Y9, Y9, Y9 \
	VPXOR Y10, Y10, Y10 \
	VPXOR Y11, Y11, Y11 \
	VPXOR Y12, Y12, Y12 \
	VPXOR Y13, Y13, Y13 \
	VPXOR Y14, Y14, Y14 \
	VPXOR Y15, Y15, Y15

#define SETUP \
	MOVQ nb+0(FP), CX \
	MOVQ a+8(FP), SI \
	MOVQ lda+16(FP), R8 \
	MOVQ as+24(FP), R11 \
	MOVQ w+32(FP), DX \
	MOVQ c+40(FP), DI \
	MOVQ ldc+48(FP), R9 \
	SHLQ $2, R9 \
	MOVQ CX, R12 \
	SHLQ $2, R12 \
	LEAQ (SI)(R8*2), R10 \
	ADDQ R8, R10 \
	LEAQ (R11)(R12*2), R13 \
	ADDQ R12, R13 \
	LEAQ (DI)(R9*2), BX \
	ADDQ R9, BX

#define BLOCK \
	ZERO \
	STEP(0, 0, 32) \
	STEP(4, 64, 96) \
	STEP(8, 128, 160) \
	STEP(12, 192, 224) \
	STEP(16, 256, 288) \
	STEP(20, 320, 352) \
	STEP(24, 384, 416) \
	STEP(28, 448, 480) \
	FIN(Y4, Y5, (R11), (DI), 32(DI)) \
	FIN(Y6, Y7, (R11)(R12*1), (DI)(R9*1), 32(DI)(R9*1)) \
	FIN(Y8, Y9, (R11)(R12*2), (DI)(R9*2), 32(DI)(R9*2)) \
	FIN(Y10, Y11, (R13), (BX), 32(BX)) \
	FIN(Y12, Y13, (R13)(R12*1), (BX)(R9*1), 32(BX)(R9*1)) \
	FIN(Y14, Y15, (R13)(R12*2), (BX)(R9*2), 32(BX)(R9*2)) \
	ADDQ $32, SI \
	ADDQ $32, R10 \
	ADDQ $4, R11 \
	ADDQ $4, R13 \
	ADDQ $640, DX

// func q8tileEVEX(nb int, a *byte, lda int, as *float32, w *byte, c *float32, ldc int)
// a: six rows of nb·32 activation bytes, lda apart; as: their scales, nb
// per row; w: one panel of nb blocks; c: six rows of 16, ldc apart.
TEXT ·q8tileEVEX(SB), NOSPLIT, $0-56
	SETUP
	TESTQ CX, CX
	JEQ edone
eloop:
	BLOCK
	DECQ CX
	JNZ eloop
edone:
	VZEROUPPER
	RET

#undef DP4
#undef DP5
#undef DP6
#undef DP7
#undef DP8
#undef DP9
#undef DP10
#undef DP11
#undef DP12
#undef DP13
#undef DP14
#undef DP15

// The same instructions in their VEX (AVX-VNNI) encoding.
#define DP4 BYTE $0xc4; BYTE $0xe2; BYTE $0x6d; BYTE $0x50; BYTE $0xe0 // vpdpbusd ymm4, ymm2, ymm0
#define DP5 BYTE $0xc4; BYTE $0xe2; BYTE $0x6d; BYTE $0x50; BYTE $0xe9 // vpdpbusd ymm5, ymm2, ymm1
#define DP6 BYTE $0xc4; BYTE $0xe2; BYTE $0x65; BYTE $0x50; BYTE $0xf0 // vpdpbusd ymm6, ymm3, ymm0
#define DP7 BYTE $0xc4; BYTE $0xe2; BYTE $0x65; BYTE $0x50; BYTE $0xf9 // vpdpbusd ymm7, ymm3, ymm1
#define DP8 BYTE $0xc4; BYTE $0x62; BYTE $0x6d; BYTE $0x50; BYTE $0xc0 // vpdpbusd ymm8, ymm2, ymm0
#define DP9 BYTE $0xc4; BYTE $0x62; BYTE $0x6d; BYTE $0x50; BYTE $0xc9 // vpdpbusd ymm9, ymm2, ymm1
#define DP10 BYTE $0xc4; BYTE $0x62; BYTE $0x65; BYTE $0x50; BYTE $0xd0 // vpdpbusd ymm10, ymm3, ymm0
#define DP11 BYTE $0xc4; BYTE $0x62; BYTE $0x65; BYTE $0x50; BYTE $0xd9 // vpdpbusd ymm11, ymm3, ymm1
#define DP12 BYTE $0xc4; BYTE $0x62; BYTE $0x6d; BYTE $0x50; BYTE $0xe0 // vpdpbusd ymm12, ymm2, ymm0
#define DP13 BYTE $0xc4; BYTE $0x62; BYTE $0x6d; BYTE $0x50; BYTE $0xe9 // vpdpbusd ymm13, ymm2, ymm1
#define DP14 BYTE $0xc4; BYTE $0x62; BYTE $0x65; BYTE $0x50; BYTE $0xf0 // vpdpbusd ymm14, ymm3, ymm0
#define DP15 BYTE $0xc4; BYTE $0x62; BYTE $0x65; BYTE $0x50; BYTE $0xf9 // vpdpbusd ymm15, ymm3, ymm1

// func q8tileVEX(nb int, a *byte, lda int, as *float32, w *byte, c *float32, ldc int)
TEXT ·q8tileVEX(SB), NOSPLIT, $0-56
	SETUP
	TESTQ CX, CX
	JEQ vdone
vloop:
	BLOCK
	DECQ CX
	JNZ vloop
vdone:
	VZEROUPPER
	RET

DATA qconst<>+0(SB)/4, $0x7fffffff  // abs mask
DATA qconst<>+4(SB)/4, $0x42fe0000  // 127
DATA qconst<>+8(SB)/4, $0x3c010204  // 1/127
DATA qconst<>+12(SB)/4, $0x80808080 // sign bit of each byte
DATA qconst<>+16(SB)/4, $0
DATA qconst<>+20(SB)/4, $4
DATA qconst<>+24(SB)/4, $1
DATA qconst<>+28(SB)/4, $5
DATA qconst<>+32(SB)/4, $2
DATA qconst<>+36(SB)/4, $6
DATA qconst<>+40(SB)/4, $3
DATA qconst<>+44(SB)/4, $7
GLOBL qconst<>(SB), RODATA|NOPTR, $48

// func quantU8avx(dst *byte, sc *float32, x *float32, nb int)
// Quantizes nb blocks of 32 values: sc[b] = max|x|/127 and dst holds
// round(x/sc[b])+128 as unsigned bytes (128 for an all-zero block).
TEXT ·quantU8avx(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ sc+8(FP), BX
	MOVQ x+16(FP), SI
	MOVQ nb+24(FP), CX
	VBROADCASTSS qconst<>+0(SB), Y8
	VBROADCASTSS qconst<>+4(SB), X9
	VBROADCASTSS qconst<>+8(SB), X10
	VPBROADCASTD qconst<>+12(SB), Y11
	VMOVDQU qconst<>+16(SB), Y12
	VXORPS X13, X13, X13
	TESTQ CX, CX
	JEQ qdone
qloop:
	VMOVUPS (SI), Y0
	VMOVUPS 32(SI), Y1
	VMOVUPS 64(SI), Y2
	VMOVUPS 96(SI), Y3
	VANDPS Y8, Y0, Y4
	VANDPS Y8, Y1, Y5
	VANDPS Y8, Y2, Y6
	VANDPS Y8, Y3, Y7
	VMAXPS Y5, Y4, Y4
	VMAXPS Y7, Y6, Y6
	VMAXPS Y6, Y4, Y4
	VEXTRACTF128 $1, Y4, X5
	VMAXPS X5, X4, X4
	VMOVHLPS X4, X4, X5
	VMAXPS X5, X4, X4
	VPSHUFD $1, X4, X5
	VMAXSS X5, X4, X4
	// scale and its inverse, 0 for a block of zeros
	VMULSS X10, X4, X5
	VMOVSS X5, (BX)
	VDIVSS X4, X9, X6
	VCMPSS $4, X13, X4, X7
	VANDPS X7, X6, X6
	VBROADCASTSS X6, Y6
	VMULPS Y6, Y0, Y0
	VMULPS Y6, Y1, Y1
	VMULPS Y6, Y2, Y2
	VMULPS Y6, Y3, Y3
	VCVTPS2DQ Y0, Y0
	VCVTPS2DQ Y1, Y1
	VCVTPS2DQ Y2, Y2
	VCVTPS2DQ Y3, Y3
	VPACKSSDW Y1, Y0, Y0
	VPACKSSDW Y3, Y2, Y2
	VPACKSSWB Y2, Y0, Y0
	VPERMD Y0, Y12, Y0
	VPXOR Y11, Y0, Y0
	VMOVDQU Y0, (DI)
	ADDQ $128, SI
	ADDQ $32, DI
	ADDQ $4, BX
	DECQ CX
	JNZ qloop
qdone:
	VZEROUPPER
	RET
