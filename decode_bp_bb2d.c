// decode_bp_bb2d.c — Proper BP decoder for 2D BB code [[N,K,D]]
// Min-sum belief propagation on the Tanner graph, array-based, handles degeneracy.
// Build: gcc -std=gnu11 -O3 -o decode_bp_bb2d decode_bp_bb2d.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define R 40
#define S 40
#define NF (R*S)
#define BB (2*NF)
#define MAX_DEG 4
#define MAX_ITER 50

uint16_t sm_deg[BB], sm_idx[BB][MAX_DEG]; // qubit -> checks
uint16_t ch_deg[NF], ch_idx[NF][MAX_DEG]; // check -> qubits

// BP messages: q2c[q][d] = msg from qubit q to its d-th check
//              c2q[c][d] = msg from check c to its d-th qubit
float q2c[BB][MAX_DEG], c2q[NF][MAX_DEG];

void build(void) {
    uint16_t gi[4]={0,2,0,2}, gj[4]={0,0,2,2};
    uint16_t bi[4], bj[4];
    for(int t=0;t<4;t++){bi[t]=(gi[t]+2)%R; bj[t]=(gj[t]+2)%S;}
    for(int u=0;u<R;u++)for(int v=0;v<S;v++){
        int k=u*S+v;
        for(int i=0;i<R;i++)for(int j=0;j<S;j++){
            int qi=i*S+j, di=(i-u+R)%R, dj=(j-v+S)%S;
            for(int t=0;t<4;t++){
                if(di==bi[t]&&dj==bj[t]){
                    int d=sm_deg[qi];
                    sm_idx[qi][d]=k; sm_deg[qi]++;
                    ch_idx[k][ch_deg[k]++]=qi;
                }
                if(di==gi[t]&&dj==gj[t]){
                    int qb=NF+qi, d=sm_deg[qb];
                    sm_idx[qb][d]=k; sm_deg[qb]++;
                    ch_idx[k][ch_deg[k]++]=qb;
                }
            }
        }
    }
}

void syndrome_of(uint8_t *err, uint8_t *syn) {
    memset(syn,0,NF);
    for(int q=0;q<BB;q++) if(err[q])
        for(int d=0;d<sm_deg[q];d++) syn[sm_idx[q][d]] ^= 1;
}

// Find the d-th neighbor index in qubit q's check list
int ch_pos_in_qubit(int c, int q) {
    for(int d=0;d<sm_deg[q];d++)
        if(sm_idx[q][d]==c) return d;
    return -1;
}

void bp_decode(uint8_t *syn, uint8_t *out, float p_err) {
    float prior = logf((1.0f-p_err)/p_err);
    int nq=BB, nc=NF;
    
    // Init messages
    for(int q=0;q<nq;q++)
        for(int d=0;d<sm_deg[q];d++)
            q2c[q][d] = prior;
    for(int c=0;c<nc;c++)
        for(int d=0;d<ch_deg[c];d++)
            c2q[c][d] = 0.0f;
    
    for(int it=0;it<MAX_ITER;it++) {
        // Check node update (min-sum)
        for(int c=0;c<nc;c++) {
            int deg = ch_deg[c];
            if(deg==0) continue;
            // Precompute total sign and min1/min2
            float total_sign = 1.0f;
            float min1=1e9f, min2=1e9f;
            for(int d=0;d<deg;d++){
                int q = ch_idx[c][d];
                int pos = ch_pos_in_qubit(c,q);
                float msg = (pos>=0) ? q2c[q][pos] : prior;
                total_sign *= (msg>=0 ? 1.0f : -1.0f);
                float a=fabsf(msg);
                if(a<min1){min2=min1;min1=a;}
                else if(a<min2) min2=a;
            }
            for(int d=0;d<deg;d++){
                int q = ch_idx[c][d];
                int pos = ch_pos_in_qubit(c,q);
                if(pos<0){c2q[c][d]=0.0f; continue;}
                float msg = q2c[q][pos];
                float sign = total_sign * (msg>=0 ? 1.0f : -1.0f);
                if(syn[c]) sign *= -1.0f; // syndrome flip
                float min_val = (fabsf(msg)<=min1+1e-6f) ? min2 : min1;
                if(isinf(min_val)) min_val=0.0f;
                c2q[c][d] = sign * min_val;
            }
        }
        // Variable node update
        for(int q=0;q<nq;q++){
            int deg = sm_deg[q];
            float total = prior;
            for(int d=0;d<deg;d++) total += c2q[sm_idx[q][d]][ch_pos_in_qubit(sm_idx[q][d],q)];
            for(int d=0;d<deg;d++){
                int c = sm_idx[q][d];
                int cpos = ch_pos_in_qubit(c,q);
                float msg = (cpos>=0) ? c2q[c][cpos] : 0.0f;
                q2c[q][d] = total - msg;
            }
        }
    }
    // Decision
    for(int q=0;q<nq;q++){
        float total = prior;
        for(int d=0;d<sm_deg[q];d++){
            int c=sm_idx[q][d];
            total += c2q[c][ch_pos_in_qubit(c,q)];
        }
        out[q] = (total < 0.0f) ? 1 : 0;
    }
}

int equiv(uint8_t *a, uint8_t *b) {
    uint8_t diff[BB], syn[NF];
    for(int i=0;i<BB;i++) diff[i]=a[i]^b[i];
    syndrome_of(diff,syn);
    for(int i=0;i<NF;i++) if(syn[i]) return 0;
    return 1;
}

int main(void) {
    srand(time(0));
    build();
    
    printf("BP Decoder for [[%d,%d,%d]] 40x40 Torus\n", BB, NF+2*R+2*S-4, R/2);
    
    int weights[]={1,2,3,5,7};
    for(int wi=0;wi<5;wi++){
        int w=weights[wi];
        int trials=200, ok=0;
        for(int t=0;t<trials;t++){
            uint8_t err[BB]; memset(err,0,BB);
            for(int i=0;i<w;){
                int q=rand()%BB;
                if(!err[q]){err[q]=1;i++;}
            }
            uint8_t syn[NF]; syndrome_of(err,syn);
            uint8_t dec[BB];
            bp_decode(syn,dec,0.001f);
            if(equiv(err,dec)) ok++;
        }
        printf("weight-%d: %3d/%3d (%5.1f%%)\n",w,ok,trials,100.0*ok/trials);
    }
    return 0;
}
