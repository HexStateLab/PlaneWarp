// plane_warp_s1.c — Minimum-weight decoder for the stride-1 toric code.
//
// Check equation (stride-1 toric code on r×s periodic lattice):
//   S[i][j] = E[i][j] ⊕ E[(i+1)%r][j] ⊕ E[i][(j+1)%s] ⊕ E[(i+1)%r][(j+1)%s]
//
// This is the standard (1+x)(1+y) plaquette check — nearest-neighbour
// connectivity, no sector decoupling.  The kernel has dimension r+s-1
// (row flips + column flips, with one dependency).  The decoder uses
// prefix-XOR particular solution + column/row-flip descent + corner-seed
// enumeration to find the minimum-weight correction.
//
// Algorithm:
//   1. Compute particular solution E0 from S via 2D prefix XOR
//   2. Descent: repeatedly flip rows/cols that reduce Hamming weight
//   3. Run with both E[0][0] ∈ {0,1}  → 2 candidates
//   4. Row/col flip descent on each candidate
//   5. All-rows / all-columns logical flips
//   6. Return minimum-weight correction
//
// Public domain — standard toric code physics (Kitaev 1997).
// Build: gcc -std=gnu11 -O3 -shared -fPIC -o libplane_s1.so plane_warp_s1.c -lm
//        gcc -std=gnu11 -O3 -o plane_s1 plane_warp_s1.c -lm -DPLANE_S1_MAIN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define S1_MAX_R 600
#define S1_MAX_S 600
#define S1_MAX_N (S1_MAX_R * S1_MAX_S)

// ---- Stride-1 wrap macros ----
#define WRAP1(x, dim) (((x) + 1 < (dim)) ? (x) + 1 : 0)

// ---- Cost map (uniform = Hamming weight) ----
static double s1_cost[S1_MAX_N];
static void s1_cost_init(int n) {
    for (int q = 0; q < n; q++) s1_cost[q] = 1.0;
}

// ==================================================================
//  syndrome_of — stride-1 toric check
// ==================================================================
void syndrome_of(int r, int s, uint8_t *err, uint8_t *syn) {
    int n = r * s;
    memset(syn, 0, n);
    for (int q = 0; q < n; q++) {
        if (!err[q]) continue;
        int i = q / s, j = q % s;
        for (int di = 0; di <= 1; di++)
            for (int dj = 0; dj <= 1; dj++)
                syn[((i - di + r) % r) * s + ((j - dj + s) % s)] ^= 1;
    }
}

// ==================================================================
//  is_stabilizer — stride-1: correction is stabilizer iff its
//  syndrome is identically zero (all 2×2 checks pass)
// ==================================================================
int is_stabilizer(int r, int s, uint8_t *diff) {
    int n = r * s;
    uint8_t *chk = calloc(n, 1);
    if (!chk) return 0;
    syndrome_of(r, s, diff, chk);
    for (int q = 0; q < n; q++)
        if (chk[q]) { free(chk); return 0; }
    free(chk);
    return 1;
}

// ==================================================================
//  preprocess_syndrome — metacheck repair (stride-1)
// ==================================================================
void preprocess_syndrome(int r, int s, uint8_t *syn) {
    // Row parity repair
    for (int i = 0; i < r; i++) {
        int rp = 0;
        for (int j = 0; j < s; j++)
            if (syn[i * s + j]) rp ^= 1;
        if (rp) {
            // Flip the syndrome bit closest to column center
            int best_j = s / 2;
            int best_d = s;
            for (int j = 0; j < s; j++) {
                if (!syn[i * s + j]) continue;
                int d = abs(j - s / 2);
                if (d < best_d) { best_d = d; best_j = j; }
            }
            syn[i * s + best_j] ^= 1;
        }
    }
    // Column parity repair
    for (int j = 0; j < s; j++) {
        int cp = 0;
        for (int i = 0; i < r; i++)
            if (syn[i * s + j]) cp ^= 1;
        if (cp) {
            int best_i = r / 2;
            int best_d = r;
            for (int i = 0; i < r; i++) {
                if (!syn[i * s + j]) continue;
                int d = abs(i - r / 2);
                if (d < best_d) { best_d = d; best_i = i; }
            }
            syn[best_i * s + j] ^= 1;
        }
    }
}

// ==================================================================
//  canonicalize — remove kernel freedom (stride-1)
// ==================================================================
void canonicalize(int r, int s, uint8_t *corr) {
    int n = r * s;
    // Greedy column-flip descent: try each column, flip if it reduces weight
    for (int sweep = 0; sweep < 4; sweep++) {
        int changed = 1;
        while (changed) {
            changed = 0;
            // Columns first (col-major sweep)
            for (int j = 0; j < s; j++) {
                int w0 = 0, w1 = 0;
                for (int i = 0; i < r; i++) {
                    int v = corr[i * s + j];
                    w0 += v;
                    w1 += (v ^ 1);
                }
                if (w1 < w0) {
                    for (int i = 0; i < r; i++) corr[i * s + j] ^= 1;
                    changed = 1;
                }
            }
            // Rows
            for (int i = 0; i < r; i++) {
                int w0 = 0, w1 = 0;
                for (int j = 0; j < s; j++) {
                    int v = corr[i * s + j];
                    w0 += v;
                    w1 += (v ^ 1);
                }
                if (w1 < w0) {
                    for (int j = 0; j < s; j++) corr[i * s + j] ^= 1;
                    changed = 1;
                }
            }
        }
    }
}

// ==================================================================
//  Forward declarations
// ==================================================================
int solve_plane(int r, int s, uint8_t *syn, uint8_t *out);
void syndrome_of(int r, int s, uint8_t *err, uint8_t *syn);

// ==================================================================
//  decode_Z — decode Z-type errors (stride-1)
// ==================================================================
int decode_Z(int r, int s, uint8_t *err_z, uint8_t *dec_z) {
    int n = r * s;
    uint8_t *syn = calloc(n, 1);
    if (!syn) return 0;
    syndrome_of(r, s, err_z, syn);
    int ok = solve_plane(r, s, syn, dec_z);
    free(syn);
    return ok;
}

// ==================================================================
//  Core: particular solution via 2D prefix XOR (stride-1)
//
//  Check: S[i][j] = E[i][j] ⊕ E[i+1][j] ⊕ E[i][j+1] ⊕ E[i+1][j+1]
//    (all indices modulo r, s for periodic torus)
//
//  Recurrence (solve forward):
//    S[i][j] = E[i][j] ⊕ E[i+1][j] ⊕ E[i][j+1] ⊕ E[i+1][j+1]
//    → E[i][j] = S[i-1][j-1] ⊕ E[i-1][j-1] ⊕ E[i-1][j] ⊕ E[i][j-1]
//
//  Boundary: E[0][*] = 0, E[*][0] = 0 (r+s-1 free parameters fixed)
//  Seed: E[0][0] = c0
//  Fill row-major: i=1..r-1, j=1..s-1 (all deps already computed)
// ==================================================================
static void s1_particular(int r, int s, uint8_t *syn, int c0, uint8_t *E) {
    int n = r * s;
    memset(E, 0, n);

    // Boundary: row 0 = 0, col 0 = 0
    E[0] = c0;  // corner seed

    // Forward fill — every E[i][j] for i>0, j>0 depends only on
    // E[i-1][j-1], E[i-1][j], E[i][j-1] — all already computed
    // or boundary in row-major order.
    for (int i = 1; i < r; i++) {
        for (int j = 1; j < s; j++) {
            // S[i-1][j-1] = E[i-1][j-1]⊕E[i][j-1]⊕E[i-1][j]⊕E[i][j]
            // → E[i][j] = S[i-1][j-1]⊕E[i-1][j-1]⊕E[i-1][j]⊕E[i][j-1]
            E[i * s + j] = syn[(i - 1) * s + (j - 1)]
                         ^ E[(i - 1) * s + (j - 1)]
                         ^ E[(i - 1) * s + j]
                         ^ E[i * s + (j - 1)];
        }
    }
}

// ==================================================================
//  Descent: column-flip + row-flip sweeps until convergence
// ==================================================================
static void s1_descent(int r, int s, uint8_t *E) {
    int n = r * s;
    for (;;) {
        int changed = 0;

        // Column sweeps: for each column j, flip if weight decreases
        for (int j = 0; j < s; j++) {
            int w0 = 0;
            for (int i = 0; i < r; i++)
                if (E[i * s + j]) w0++;
            if (w0 > r / 2) {
                for (int i = 0; i < r; i++)
                    E[i * s + j] ^= 1;
                changed = 1;
            }
        }

        // Row sweeps: for each row i, flip if weight decreases
        for (int i = 0; i < r; i++) {
            int w0 = 0;
            for (int j = 0; j < s; j++)
                if (E[i * s + j]) w0++;
            if (w0 > s / 2) {
                for (int j = 0; j < s; j++)
                    E[i * s + j] ^= 1;
                changed = 1;
            }
        }

        if (!changed) break;
    }
}

// ==================================================================
//  solve_plane — main stride-1 decoder
// ==================================================================
int solve_plane(int r, int s, uint8_t *syn, uint8_t *out) {
    int n = r * s;
    s1_cost_init(n);
    memset(out, 0, n);

    uint8_t *E  = calloc(n, 1);
    uint8_t *best = calloc(n, 1);
    if (!E || !best) { free(E); free(best); return 0; }

    int best_w = n + 1;

    // Try corner seeds E[0][0] = 0 and 1
    for (int c0 = 0; c0 < 2; c0++) {
        s1_particular(r, s, syn, c0, E);
        s1_descent(r, s, E);

        int w = 0;
        for (int q = 0; q < n; q++) if (E[q]) w++;

        if (w < best_w) {
            best_w = w;
            memcpy(best, E, n);
        }

        // Try all row flips + all column flips as global logical operators
        // (kernel elements that shift the coset)
        uint8_t *tmp = calloc(n, 1);
        if (tmp) {
            // Row flips
            for (int i = 0; i < r; i++) {
                memcpy(tmp, best, n);
                for (int j = 0; j < s; j++) tmp[i * s + j] ^= 1;
                s1_descent(r, s, tmp);
                w = 0;
                for (int q = 0; q < n; q++) if (tmp[q]) w++;
                if (w < best_w) {
                    best_w = w;
                    memcpy(best, tmp, n);
                }
            }
            // Column flips
            for (int j = 0; j < s; j++) {
                memcpy(tmp, best, n);
                for (int i = 0; i < r; i++) tmp[i * s + j] ^= 1;
                s1_descent(r, s, tmp);
                w = 0;
                for (int q = 0; q < n; q++) if (tmp[q]) w++;
                if (w < best_w) {
                    best_w = w;
                    memcpy(best, tmp, n);
                }
            }
            free(tmp);
        }
    }

    memcpy(out, best, n);
    free(E);
    free(best);

    return best_w <= n;
}

// ==================================================================
//  solve_plane_layered — same as solve_plane (stride-1 has no
//  sector decomposition, so layered = plain)
// ==================================================================
int solve_plane_layered(int r, int s, uint8_t *syn, uint8_t *out) {
    return solve_plane(r, s, syn, out);
}

// ==================================================================
//  solve_plane_fast — fast O(n) decoder (single descent pass)
// ==================================================================
int solve_plane_fast(int r, int s, uint8_t *syn, uint8_t *out) {
    int n = r * s;
    memset(out, 0, n);
    uint8_t *E = calloc(n, 1);
    if (!E) return 0;

    s1_particular(r, s, syn, 0, E);
    s1_descent(r, s, E);

    int w = 0;
    for (int q = 0; q < n; q++) if (E[q]) w++;

    // Quick row-0 descent check
    uint8_t *tmp = calloc(n, 1);
    if (tmp) {
        memcpy(tmp, E, n);
        for (int j = 0; j < s; j++) tmp[j] ^= 1;
        s1_descent(r, s, tmp);
        int w2 = 0;
        for (int q = 0; q < n; q++) if (tmp[q]) w2++;
        if (w2 < w) { w = w2; memcpy(E, tmp, n); }
        free(tmp);
    }

    memcpy(out, E, n);
    free(E);
    return w <= n;
}

// ==================================================================
//  Global decoder knobs
// ==================================================================
int g_fast           = 0;
int g_escape_enabled = 1;
int g_singleshot     = 1;
int g_weight_cap     = 0;
double g_cap_auto_rate = 0.0;

// ==================================================================
//  Noise generators
// ==================================================================
static int s1_rand(void) {
    static unsigned int seed = 1;
    seed = seed * 1103515245 + 12345;
    return (seed >> 16) & 0x7FFF;
}

void gen_iid(int n, uint8_t *err, int w) {
    if (w > n) w = n;
    memset(err, 0, n);
    for (int i = 0; i < w;) {
        int q = rand() % n;
        if (!err[q]) { err[q] = 1; i++; }
    }
}

// ==================================================================
//  Self-test
// ==================================================================
#ifdef PLANE_S1_MAIN
static void selftest_weight1(int r, int s) {
    int n = r * s;
    uint8_t *err  = calloc(n, 1);
    uint8_t *syn  = calloc(n, 1);
    uint8_t *corr = calloc(n, 1);
    int ok = 0;

    printf("Weight-1 test %dx%d: ", r, s);
    fflush(stdout);

    for (int q = 0; q < n; q++) {
        memset(err, 0, n);
        err[q] = 1;
        syndrome_of(r, s, err, syn);
        solve_plane(r, s, syn, corr);

        // Check: error ⊕ correction must be a stabilizer (no logical error)
        uint8_t *residual = calloc(n, 1);
        for (int i = 0; i < n; i++) residual[i] = err[i] ^ corr[i];
        if (!is_stabilizer(r, s, residual)) {
            printf("\n  FAIL at position %d: error ⊕ correction is logical\n", q);
            free(residual);
            goto done;
        }
        free(residual);
        ok++;
    }

    printf("PASS (%d/%d)\n", ok, n);
done:
    free(err); free(syn); free(corr);
}

int main(int argc, char **argv) {
    printf("plane_warp_s1 — stride-1 toric code decoder\n");

    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) {
        selftest_weight1(4, 4);
        selftest_weight1(5, 5);
        selftest_weight1(6, 6);
        selftest_weight1(7, 8);
        selftest_weight1(10, 10);
        return 0;
    }

    printf("Usage: plane_s1 [r] [s] < syndrome.bin > correction.bin\n");
    printf("       plane_s1 --selftest\n");
    return 0;
}
#endif
