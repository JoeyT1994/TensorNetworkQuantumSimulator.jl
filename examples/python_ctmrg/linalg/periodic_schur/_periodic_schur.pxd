cdef public api int periodic_schur_active_size(
    const unsigned char* active_cols,
    int period,
    int m,
) noexcept nogil

cdef public api int periodic_schur_active_min_size(
    const unsigned char* active_cols,
    int period,
    int m,
) noexcept nogil

cdef public api int compute_periodic_schur_eigenvalues_D(
    const double* T,
    int period,
    int input_width,
    int n,
    void* eigenvalues,
) noexcept nogil

cdef public api int compute_periodic_schur_eigenvalues_Z(
    const void* T,
    int period,
    int input_width,
    int n,
    void* eigenvalues,
) noexcept nogil

cdef public api int compute_periodic_schur_active_D(
    const double* H,
    const unsigned char* active_cols,
    double schur_deflation_tol,
    int period,
    int m,
    int n,
    int output_width,
    double* T,
    double* Z,
    double* wr,
    double* wi,
) noexcept nogil

cdef public api int compute_periodic_schur_active_CRed_D(
    const double* H,
    const unsigned char* active_cols,
    double schur_deflation_tol,
    int period,
    int m,
    int capacity,
    int n,
    int output_width,
    double* T,
    double* Z,
    double* wr,
    double* wi,
) noexcept nogil

cdef public api int compute_periodic_schur_active_scaled_D(
    const double* H,
    const unsigned char* active_cols,
    const double* scale_tol,
    double schur_deflation_tol,
    int period,
    int m,
    int n,
    int output_width,
    double* T,
    double* Z,
    double* wr,
    double* wi,
) noexcept nogil

cdef public api int compute_periodic_schur_active_Z(
    const void* H,
    const unsigned char* active_cols,
    int period,
    int m,
    int n,
    int output_width,
    void* T,
    void* Z,
    void* alpha,
    void* beta,
    int* scale,
) noexcept nogil

cdef public api int compute_periodic_schur_active_CRed_Z(
    const void* H,
    const unsigned char* active_cols,
    int period,
    int m,
    int capacity,
    int n,
    int output_width,
    void* T,
    void* Z,
    void* alpha,
    void* beta,
    int* scale,
) noexcept nogil

cdef public api int compute_periodic_schur_active_scaled_Z(
    const void* H,
    const unsigned char* active_cols,
    const double* scale_tol,
    int period,
    int m,
    int n,
    int output_width,
    void* T,
    void* Z,
    void* alpha,
    void* beta,
    int* scale,
) noexcept nogil

cdef public api int compute_reordered_periodic_schur_D(
    const double* T,
    const double* Z,
    const unsigned char* select,
    int period,
    int m,
    int n,
    double tol,
    double* T_out,
    double* Z_out,
) noexcept nogil

cdef public api int compute_reordered_periodic_schur_Z(
    const void* T,
    const void* Z,
    const unsigned char* select,
    int period,
    int m,
    int n,
    double tol,
    void* T_out,
    void* Z_out,
) noexcept nogil

cdef api int slicot_mb03ke_D(
    int trana,
    int tranb,
    int isgn,
    int period,
    int m,
    int n,
    const int* signs,
    const double* A,
    const double* B,
    double* C,
    double* scale,
    double* work,
    int lwork,
) noexcept nogil
