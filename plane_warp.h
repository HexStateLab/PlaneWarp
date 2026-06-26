#ifndef PLANE_WARP_H
#define PLANE_WARP_H

#include <stdint.h>

#define PW_MAX_R 600
#define PW_MAX_S 600
#define PW_MAX_N (PW_MAX_R * PW_MAX_S)

// Decode a single syndrome using the full alternating-optimisation solver.
//   syn  — r*s syndrome bytes (0/1), row-major.
//   out  — receives r*s correction bytes (0/1), row-major.
// Returns 1 on success, 0 if the decoder abstained (weight cap exceeded).
int solve_plane(int r, int s, uint8_t *syn, uint8_t *out);

// Layered decoder — explicit recursion through sub-lattice blocks.
// Same interface as solve_plane.
int solve_plane_layered(int r, int s, uint8_t *syn, uint8_t *out);

// Fast O(n) adaptive-corner solver — less accurate but ~10× faster.
int solve_plane_fast(int r, int s, uint8_t *syn, uint8_t *out);

// Compute syndrome = H * error for the (1+x²)(1+y²) check.
void syndrome_of(int r, int s, uint8_t *err, uint8_t *syn);

// Preprocess a syndrome in-place: repair measurement faults via metachecks.
void preprocess_syndrome(int r, int s, uint8_t *syn);

// Canonicalize a correction in-place: removes kernel freedom so every
// syndrome maps to a unique representative.
void canonicalize(int r, int s, uint8_t *corr);

// Check whether a correction is a pure stabiliser (zero logical action).
int is_stabilizer(int r, int s, uint8_t *diff);

// Decode Z-type errors (shifted syndrome).
int decode_Z(int r, int s, uint8_t *err_z, uint8_t *dec_z);

// Global decoder knobs (set before calling any decode function):
extern int g_fast;
extern int g_escape_enabled;
extern int g_singleshot;
extern int g_weight_cap;
extern double g_cap_auto_rate;

#endif // PLANE_WARP_H
