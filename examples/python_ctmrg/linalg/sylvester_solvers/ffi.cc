#include <cstdint>
#include <string>

#include "xla/ffi/api/ffi.h"

int sylvester_compressed_schur_gmres_D(
    const double* H, const double* w, const double* v,
    const double* residual_r, const unsigned char* block_2x2_start,
    int n_krylov, int d_block, int upper, int tpqrt_block_size,
    double* x, double* error);

int sylvester_compressed_periodic_dense_gmres_D(
    const double* H, const double* w, const double* v,
    const double* residual_r, const unsigned char* block_2x2_start,
    const unsigned char* active_cols,
    int period, int n_krylov, int d_block, int rank,
    double* x, double* error);

int sylvester_compressed_periodic_dense_gmres_Z(
    const void* H, const void* w, const void* v, const void* residual_r,
    const unsigned char* active_cols,
    int period, int n_krylov, int d_block, int rank,
    void* x, double* error);

int sylvester_compressed_periodic_schur_galerkin_Z(
    const void* H, const void* w, const void* v, const void* residual_r,
    const double* scale_tol,
    const unsigned char* active_cols,
    int period, int n_krylov, int d_block, int rank,
    void* x, double* error);

int sylvester_compressed_periodic_schur_gmres_D(
    const double* H, const double* w, const double* v,
    const double* residual_r, const double* scale_tol,
    const unsigned char* block_2x2_start,
    const unsigned char* active_cols,
    int period, int n_krylov, int d_block, int rank,
    double* x, double* error);

int sylvester_compressed_periodic_schur_galerkin_D(
    const double* H, const double* w, const double* v,
    const double* residual_r, const double* scale_tol,
    const unsigned char* block_2x2_start,
    const unsigned char* active_cols,
    int period, int n_krylov, int d_block, int rank, int use_mb03ke,
    double* x, double* error);

int sylvester_compressed_periodic_schur_gmres_Z(
    const void* H, const void* w, const void* v, const void* residual_r,
    const double* scale_tol,
    const unsigned char* active_cols,
    int period, int n_krylov, int d_block, int rank,
    void* x, double* error);

namespace ffi = xla::ffi;
using F64R1 = ffi::Buffer<ffi::F64, 1>;
using F64R2 = ffi::Buffer<ffi::F64, 2>;
using F64R3 = ffi::Buffer<ffi::F64, 3>;
using C128R3 = ffi::Buffer<ffi::C128, 3>;
using PredR1 = ffi::Buffer<ffi::PRED, 1>;
using PredR2 = ffi::Buffer<ffi::PRED, 2>;
using S32R0 = ffi::Buffer<ffi::S32, 0>;

ffi::Error SylvesterCompressedSchurGmresRealImpl(
    bool upper, std::int64_t tpqrt_block_size,
    F64R2 H,
    F64R2 w,
    F64R2 v,
    F64R2 residual_r,
    ffi::Buffer<ffi::PRED, 1> block_2x2_start,
    ffi::ResultBuffer<ffi::F64, 2> x,
    ffi::ResultBuffer<ffi::F64, 1> error) {
  const int n_krylov = static_cast<int>(H.dimensions()[0]);
  const int d_block = static_cast<int>(w.dimensions()[0]);
  int info = sylvester_compressed_schur_gmres_D(
      H.typed_data(), w.typed_data(), v.typed_data(),
      residual_r.typed_data(),
      reinterpret_cast<const unsigned char*>(block_2x2_start.typed_data()),
      n_krylov, d_block, upper, static_cast<int>(tpqrt_block_size),
      x->typed_data(), error->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "real compressed Sylvester solver failed with info="
        + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SylvesterCompressedSchurGmresRealF64,
    SylvesterCompressedSchurGmresRealImpl,
    ffi::Ffi::Bind()
        .Attr<bool>("upper")
        .Attr<std::int64_t>("tpqrt_block_size")
        .Arg<F64R2>()
        .Arg<F64R2>()
        .Arg<F64R2>()
        .Arg<F64R2>()
        .Arg<PredR1>()
        .Ret<F64R2>()
        .Ret<F64R1>());

ffi::Error SylvesterCompressedPeriodicDenseGmresRealImpl(
    F64R3 H,
    F64R3 w,
    F64R3 v,
    F64R3 residual_r,
    ffi::Buffer<ffi::PRED, 1> block_2x2_start,
    ffi::Buffer<ffi::PRED, 2> active_cols,
    ffi::Buffer<ffi::S32, 0> rank,
    ffi::ResultBuffer<ffi::F64, 3> x,
    ffi::ResultBuffer<ffi::F64, 2> error) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int n_krylov = static_cast<int>(H.dimensions()[1]);
  const int d_block = static_cast<int>(w.dimensions()[1]);
  int info = sylvester_compressed_periodic_dense_gmres_D(
      H.typed_data(), w.typed_data(), v.typed_data(),
      residual_r.typed_data(),
      reinterpret_cast<const unsigned char*>(block_2x2_start.typed_data()),
      reinterpret_cast<const unsigned char*>(active_cols.typed_data()),
      period, n_krylov, d_block, rank.typed_data()[0],
      x->typed_data(), error->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "real periodic compressed Sylvester solver failed with info="
        + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SylvesterCompressedPeriodicDenseGmresRealF64,
    SylvesterCompressedPeriodicDenseGmresRealImpl,
    ffi::Ffi::Bind()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<PredR1>()
        .Arg<PredR2>()
        .Arg<S32R0>()
        .Ret<F64R3>()
        .Ret<F64R2>());

ffi::Error SylvesterCompressedPeriodicDenseGmresComplexImpl(
    C128R3 H,
    C128R3 w,
    C128R3 v,
    C128R3 residual_r,
    ffi::Buffer<ffi::PRED, 2> active_cols,
    ffi::Buffer<ffi::S32, 0> rank,
    ffi::ResultBuffer<ffi::C128, 3> x,
    ffi::ResultBuffer<ffi::F64, 2> error) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int n_krylov = static_cast<int>(H.dimensions()[1]);
  const int d_block = static_cast<int>(w.dimensions()[1]);
  int info = sylvester_compressed_periodic_dense_gmres_Z(
      H.typed_data(), w.typed_data(), v.typed_data(),
      residual_r.typed_data(),
      reinterpret_cast<const unsigned char*>(active_cols.typed_data()),
      period, n_krylov, d_block, rank.typed_data()[0],
      x->typed_data(), error->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "complex periodic compressed Sylvester solver failed with info="
        + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SylvesterCompressedPeriodicDenseGmresComplexC128,
    SylvesterCompressedPeriodicDenseGmresComplexImpl,
    ffi::Ffi::Bind()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<PredR2>()
        .Arg<S32R0>()
        .Ret<C128R3>()
        .Ret<F64R2>());

ffi::Error SylvesterCompressedPeriodicSchurGalerkinComplexImpl(
    C128R3 H,
    C128R3 w,
    C128R3 v,
    C128R3 residual_r,
    F64R1 scale_tol,
    ffi::Buffer<ffi::PRED, 2> active_cols,
    ffi::Buffer<ffi::S32, 0> rank,
    ffi::ResultBuffer<ffi::C128, 3> x,
    ffi::ResultBuffer<ffi::F64, 2> error) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int n_krylov = static_cast<int>(H.dimensions()[1]);
  const int d_block = static_cast<int>(w.dimensions()[1]);
  int info = sylvester_compressed_periodic_schur_galerkin_Z(
      H.typed_data(), w.typed_data(), v.typed_data(),
      residual_r.typed_data(),
      scale_tol.typed_data(),
      reinterpret_cast<const unsigned char*>(active_cols.typed_data()),
      period, n_krylov, d_block, rank.typed_data()[0],
      x->typed_data(), error->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "complex periodic Galerkin Sylvester solver failed with info="
        + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SylvesterCompressedPeriodicSchurGalerkinComplexC128,
    SylvesterCompressedPeriodicSchurGalerkinComplexImpl,
    ffi::Ffi::Bind()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<F64R1>()
        .Arg<PredR2>()
        .Arg<S32R0>()
        .Ret<C128R3>()
        .Ret<F64R2>());

ffi::Error SylvesterCompressedPeriodicSchurGmresRealImpl(
    F64R3 H,
    F64R3 w,
    F64R3 v,
    F64R3 residual_r,
    F64R1 scale_tol,
    ffi::Buffer<ffi::PRED, 1> block_2x2_start,
    ffi::Buffer<ffi::PRED, 2> active_cols,
    ffi::Buffer<ffi::S32, 0> rank,
    ffi::ResultBuffer<ffi::F64, 3> x,
    ffi::ResultBuffer<ffi::F64, 2> error) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int n_krylov = static_cast<int>(H.dimensions()[1]);
  const int d_block = static_cast<int>(w.dimensions()[1]);
  int info = sylvester_compressed_periodic_schur_gmres_D(
      H.typed_data(), w.typed_data(), v.typed_data(),
      residual_r.typed_data(),
      scale_tol.typed_data(),
      reinterpret_cast<const unsigned char*>(block_2x2_start.typed_data()),
      reinterpret_cast<const unsigned char*>(active_cols.typed_data()),
      period, n_krylov, d_block, rank.typed_data()[0],
      x->typed_data(), error->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "real periodic-Schur compressed Sylvester solver failed with info="
        + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SylvesterCompressedPeriodicSchurGmresRealF64,
    SylvesterCompressedPeriodicSchurGmresRealImpl,
    ffi::Ffi::Bind()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<F64R1>()
        .Arg<PredR1>()
        .Arg<PredR2>()
        .Arg<S32R0>()
        .Ret<F64R3>()
        .Ret<F64R2>());

ffi::Error SylvesterCompressedPeriodicSchurGalerkinRealImpl(
    bool use_mb03ke,
    F64R3 H,
    F64R3 w,
    F64R3 v,
    F64R3 residual_r,
    F64R1 scale_tol,
    ffi::Buffer<ffi::PRED, 1> block_2x2_start,
    ffi::Buffer<ffi::PRED, 2> active_cols,
    ffi::Buffer<ffi::S32, 0> rank,
    ffi::ResultBuffer<ffi::F64, 3> x,
    ffi::ResultBuffer<ffi::F64, 2> error) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int n_krylov = static_cast<int>(H.dimensions()[1]);
  const int d_block = static_cast<int>(w.dimensions()[1]);
  int info = sylvester_compressed_periodic_schur_galerkin_D(
      H.typed_data(), w.typed_data(), v.typed_data(),
      residual_r.typed_data(),
      scale_tol.typed_data(),
      reinterpret_cast<const unsigned char*>(block_2x2_start.typed_data()),
      reinterpret_cast<const unsigned char*>(active_cols.typed_data()),
      period, n_krylov, d_block, rank.typed_data()[0], use_mb03ke,
      x->typed_data(), error->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "real periodic-Schur Galerkin Sylvester solver failed with info="
        + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SylvesterCompressedPeriodicSchurGalerkinRealF64,
    SylvesterCompressedPeriodicSchurGalerkinRealImpl,
    ffi::Ffi::Bind()
        .Attr<bool>("use_mb03ke")
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<F64R1>()
        .Arg<PredR1>()
        .Arg<PredR2>()
        .Arg<S32R0>()
        .Ret<F64R3>()
        .Ret<F64R2>());

ffi::Error SylvesterCompressedPeriodicSchurGmresComplexImpl(
    C128R3 H,
    C128R3 w,
    C128R3 v,
    C128R3 residual_r,
    F64R1 scale_tol,
    ffi::Buffer<ffi::PRED, 2> active_cols,
    ffi::Buffer<ffi::S32, 0> rank,
    ffi::ResultBuffer<ffi::C128, 3> x,
    ffi::ResultBuffer<ffi::F64, 2> error) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int n_krylov = static_cast<int>(H.dimensions()[1]);
  const int d_block = static_cast<int>(w.dimensions()[1]);
  int info = sylvester_compressed_periodic_schur_gmres_Z(
      H.typed_data(), w.typed_data(), v.typed_data(),
      residual_r.typed_data(),
      scale_tol.typed_data(),
      reinterpret_cast<const unsigned char*>(active_cols.typed_data()),
      period, n_krylov, d_block, rank.typed_data()[0],
      x->typed_data(), error->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "complex periodic-Schur compressed Sylvester solver failed with info="
        + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SylvesterCompressedPeriodicSchurGmresComplexC128,
    SylvesterCompressedPeriodicSchurGmresComplexImpl,
    ffi::Ffi::Bind()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<F64R1>()
        .Arg<PredR2>()
        .Arg<S32R0>()
        .Ret<C128R3>()
        .Ret<F64R2>());
