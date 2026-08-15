#ifndef PERIODIC_SCHUR_SLICOT_PERIODIC_C_API_H_
#define PERIODIC_SCHUR_SLICOT_PERIODIC_C_API_H_

#include <complex>

extern "C" {

void mb03vd_(
    int* n, int* p, int* ilo, int* ihi,
    double* a, int* lda1, int* lda2,
    double* tau, int* ldtau, double* dwork, int* info);

void mb03vy_(
    int* n, int* p, int* ilo, int* ihi,
    double* a, int* lda1, int* lda2,
    double* tau, int* ldtau,
    double* dwork, int* ldwork, int* info);

void mb03wd_(
    char* job, char* compz,
    int* n, int* p, int* ilo, int* ihi, int* iloz, int* ihiz,
    double* h, int* ldh1, int* ldh2,
    double* z, int* ldz1, int* ldz2,
    double* wr, double* wi,
    double* dwork, int* ldwork, int* info);

void mb03wx_(
    int* n, int* p,
    double* t, int* ldt1, int* ldt2,
    double* wr, double* wi, int* info);

void mb03kd_(
    char* compq, int* whichq, char* strong,
    int* k, int* nc, int* kschur,
    int* n, int* ni, int* signs, int* select,
    double* t, int* ldt, int* ixt,
    double* q, int* ldq, int* ixq,
    int* m, double* tol,
    int* iwork, double* dwork, int* ldwork, int* info);

void mb03ke_(
    int* trana, int* tranb, int* isgn,
    int* k, int* m, int* n,
    double* prec, double* smin, int* signs,
    double* a, double* b, double* c, double* scale,
    double* dwork, int* ldwork, int* info);

void mb03bz_(
    char* job, char* compq,
    int* k, int* n, int* ilo, int* ihi, int* signs,
    std::complex<double>* a, int* lda1, int* lda2,
    std::complex<double>* q, int* ldq1, int* ldq2,
    std::complex<double>* alpha, std::complex<double>* beta, int* scale,
    double* dwork, int* ldwork,
    std::complex<double>* zwork, int* lzwork, int* info);

}  // extern "C"

#endif  // PERIODIC_SCHUR_SLICOT_PERIODIC_C_API_H_
