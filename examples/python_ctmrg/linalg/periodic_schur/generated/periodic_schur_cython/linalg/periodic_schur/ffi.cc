#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

extern "C" int periodic_schur_active_size(
    const unsigned char* active_cols, int period, int m);

extern "C" int periodic_schur_active_min_size(
    const unsigned char* active_cols, int period, int m);

extern "C" int compute_periodic_schur_active_D(
    const double* H, const unsigned char* active_cols,
    int period, int m, int n, int output_width,
    double* T, double* Z, double* wr, double* wi);

extern "C" int compute_periodic_schur_active_CRed_D(
    const double* H, const unsigned char* active_cols,
    int period, int m, int capacity, int n, int output_width,
    double* T, double* Z, double* wr, double* wi);

extern "C" int compute_periodic_schur_active_Z(
    const void* H, const unsigned char* active_cols,
    int period, int m, int n, int output_width,
    void* T, void* Z, void* alpha, void* beta, int* scale);

extern "C" int compute_periodic_schur_active_CRed_Z(
    const void* H, const unsigned char* active_cols,
    int period, int m, int capacity, int n, int output_width,
    void* T, void* Z, void* alpha, void* beta, int* scale);

extern "C" int compute_reordered_periodic_schur_D(
    const double* T, const double* Z, const unsigned char* select,
    int period, int m, int n, double tol, double* T_out, double* Z_out);

extern "C" int compute_reordered_periodic_schur_Z(
    const void* T, const void* Z, const unsigned char* select,
    int period, int m, int n, double tol, void* T_out, void* Z_out);

namespace ffi = xla::ffi;

using F64R0 = ffi::Buffer<ffi::F64, 0>;
using F64R1 = ffi::Buffer<ffi::F64, 1>;
using F64R3 = ffi::Buffer<ffi::F64, 3>;
using C128R1 = ffi::Buffer<ffi::C128, 1>;
using C128R3 = ffi::Buffer<ffi::C128, 3>;
using PredR1 = ffi::Buffer<ffi::PRED, 1>;
using PredR2 = ffi::Buffer<ffi::PRED, 2>;
using S32R0 = ffi::Buffer<ffi::S32, 0>;
using S32R1 = ffi::Buffer<ffi::S32, 1>;

// Run the real R-oriented NRed driver and retain static full-size outputs.
ffi::Error PeriodicSchurActiveRealImpl(
    F64R3 H,
    PredR2 active_cols,
    ffi::ResultBuffer<ffi::F64, 3> T,
    ffi::ResultBuffer<ffi::F64, 3> Z,
    ffi::ResultBuffer<ffi::F64, 1> wr,
    ffi::ResultBuffer<ffi::F64, 1> wi,
    ffi::ResultBuffer<ffi::S32, 0> active_size) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int m = static_cast<int>(H.dimensions()[1]);
  if (H.dimensions()[2] != m ||
      active_cols.dimensions()[0] != period ||
      active_cols.dimensions()[1] != m) {
    return ffi::Error::InvalidArgument(
        "H and active_cols must have shapes (period,m,m) and (period,m)");
  }
  const auto* active =
      reinterpret_cast<const unsigned char*>(active_cols.typed_data());
  const int n = periodic_schur_active_size(active, period, m);
  active_size->typed_data()[0] = n;

  const int info = compute_periodic_schur_active_D(
      H.typed_data(), active, period, m, n, m,
      T->typed_data(), Z->typed_data(), wr->typed_data(), wi->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "real periodic Schur failed with info=" + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PeriodicSchurActiveRealF64,
    PeriodicSchurActiveRealImpl,
    ffi::Ffi::Bind()
        .Arg<F64R3>()
        .Arg<PredR2>()
        .Ret<F64R3>()
        .Ret<F64R3>()
        .Ret<F64R1>()
        .Ret<F64R1>()
        .Ret<S32R0>());

// Run the real eig-only CRed driver and retain static full-size outputs.
ffi::Error PeriodicSchurActiveRealCRedImpl(
    F64R3 H,
    PredR2 active_cols,
    ffi::ResultBuffer<ffi::F64, 3> T,
    ffi::ResultBuffer<ffi::F64, 3> Z,
    ffi::ResultBuffer<ffi::F64, 1> wr,
    ffi::ResultBuffer<ffi::F64, 1> wi,
    ffi::ResultBuffer<ffi::S32, 0> active_size) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int m = static_cast<int>(H.dimensions()[1]);
  if (H.dimensions()[2] != m ||
      active_cols.dimensions()[0] != period ||
      active_cols.dimensions()[1] != m) {
    return ffi::Error::InvalidArgument(
        "H and active_cols must have shapes (period,m,m) and (period,m)");
  }
  const auto* active =
      reinterpret_cast<const unsigned char*>(active_cols.typed_data());
  const int capacity = periodic_schur_active_size(active, period, m);
  const int n = periodic_schur_active_min_size(active, period, m);
  active_size->typed_data()[0] = n;

  const int info = compute_periodic_schur_active_CRed_D(
      H.typed_data(), active, period, m, capacity, n, m,
      T->typed_data(), Z->typed_data(), wr->typed_data(), wi->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "real periodic CRed Schur failed with info=" + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PeriodicSchurActiveRealCRedF64,
    PeriodicSchurActiveRealCRedImpl,
    ffi::Ffi::Bind()
        .Arg<F64R3>()
        .Arg<PredR2>()
        .Ret<F64R3>()
        .Ret<F64R3>()
        .Ret<F64R1>()
        .Ret<F64R1>()
        .Ret<S32R0>());

// Run the complex R-oriented NRed driver and retain static full-size outputs.
ffi::Error PeriodicSchurActiveComplexImpl(
    C128R3 H,
    PredR2 active_cols,
    ffi::ResultBuffer<ffi::C128, 3> T,
    ffi::ResultBuffer<ffi::C128, 3> Z,
    ffi::ResultBuffer<ffi::C128, 1> alpha,
    ffi::ResultBuffer<ffi::C128, 1> beta,
    ffi::ResultBuffer<ffi::S32, 1> scale,
    ffi::ResultBuffer<ffi::S32, 0> active_size) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int m = static_cast<int>(H.dimensions()[1]);
  if (H.dimensions()[2] != m ||
      active_cols.dimensions()[0] != period ||
      active_cols.dimensions()[1] != m) {
    return ffi::Error::InvalidArgument(
        "H and active_cols must have shapes (period,m,m) and (period,m)");
  }
  const auto* active =
      reinterpret_cast<const unsigned char*>(active_cols.typed_data());
  const int n = periodic_schur_active_size(active, period, m);
  active_size->typed_data()[0] = n;

  const int info = compute_periodic_schur_active_Z(
      H.typed_data(), active, period, m, n, m,
      T->typed_data(), Z->typed_data(),
      alpha->typed_data(), beta->typed_data(), scale->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "complex periodic Schur failed with info=" + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PeriodicSchurActiveComplexC128,
    PeriodicSchurActiveComplexImpl,
    ffi::Ffi::Bind()
        .Arg<C128R3>()
        .Arg<PredR2>()
        .Ret<C128R3>()
        .Ret<C128R3>()
        .Ret<C128R1>()
        .Ret<C128R1>()
        .Ret<S32R1>()
        .Ret<S32R0>());

// Run the complex eig-only CRed driver and retain static full-size outputs.
ffi::Error PeriodicSchurActiveComplexCRedImpl(
    C128R3 H,
    PredR2 active_cols,
    ffi::ResultBuffer<ffi::C128, 3> T,
    ffi::ResultBuffer<ffi::C128, 3> Z,
    ffi::ResultBuffer<ffi::C128, 1> alpha,
    ffi::ResultBuffer<ffi::C128, 1> beta,
    ffi::ResultBuffer<ffi::S32, 1> scale,
    ffi::ResultBuffer<ffi::S32, 0> active_size) {
  const int period = static_cast<int>(H.dimensions()[0]);
  const int m = static_cast<int>(H.dimensions()[1]);
  if (H.dimensions()[2] != m ||
      active_cols.dimensions()[0] != period ||
      active_cols.dimensions()[1] != m) {
    return ffi::Error::InvalidArgument(
        "H and active_cols must have shapes (period,m,m) and (period,m)");
  }
  const auto* active =
      reinterpret_cast<const unsigned char*>(active_cols.typed_data());
  const int capacity = periodic_schur_active_size(active, period, m);
  const int n = periodic_schur_active_min_size(active, period, m);
  active_size->typed_data()[0] = n;

  const int info = compute_periodic_schur_active_CRed_Z(
      H.typed_data(), active, period, m, capacity, n, m,
      T->typed_data(), Z->typed_data(),
      alpha->typed_data(), beta->typed_data(), scale->typed_data());
  if (info != 0) {
    return ffi::Error::Internal(
        "complex periodic CRed Schur failed with info=" + std::to_string(info));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PeriodicSchurActiveComplexCRedC128,
    PeriodicSchurActiveComplexCRedImpl,
    ffi::Ffi::Bind()
        .Arg<C128R3>()
        .Arg<PredR2>()
        .Ret<C128R3>()
        .Ret<C128R3>()
        .Ret<C128R1>()
        .Ret<C128R1>()
        .Ret<S32R1>()
        .Ret<S32R0>());

// Reorder an R-oriented real periodic Schur form.
ffi::Error PeriodicSchurReorderRealImpl(
    F64R3 T,
    F64R3 Z,
    PredR1 select,
    S32R0 schur_size,
    F64R0 tol,
    ffi::ResultBuffer<ffi::F64, 3> T_out,
    ffi::ResultBuffer<ffi::F64, 3> Z_out) {
  const int period = static_cast<int>(T.dimensions()[0]);
  const int width = static_cast<int>(T.dimensions()[1]);
  const int m = static_cast<int>(Z.dimensions()[1]);
  const int n = schur_size.typed_data()[0];
  if (T.dimensions()[2] != width ||
      Z.dimensions()[0] != period ||
      Z.dimensions()[2] != width ||
      select.dimensions()[0] != width ||
      n < 0 || n > width) {
    return ffi::Error::InvalidArgument(
        "T, Z, and select must share a static width containing schur_size");
  }
  const auto* selected =
      reinterpret_cast<const unsigned char*>(select.typed_data());
  std::vector<double> T_live(period * n * n);
  std::vector<double> Z_live(period * m * n);
  std::vector<unsigned char> select_live(n);
  std::vector<double> T_reordered(period * n * n);
  std::vector<double> Z_reordered(period * m * n);
  for (int k = 0; k < period; ++k) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        T_live[(k * n + i) * n + j] =
            T.typed_data()[(k * width + i) * width + j];
      }
    }
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        Z_live[(k * m + i) * n + j] =
            Z.typed_data()[(k * m + i) * width + j];
      }
    }
  }
  for (int j = 0; j < n; ++j) {
    select_live[j] = selected[j];
  }
  const int info = compute_reordered_periodic_schur_D(
      T_live.data(), Z_live.data(), select_live.data(),
      period, m, n, tol.typed_data()[0],
      T_reordered.data(), Z_reordered.data());
  if (info != 0) {
    return ffi::Error::Internal(
        "real periodic Schur reordering failed with info=" +
        std::to_string(info));
  }
  std::fill_n(T_out->typed_data(), period * width * width, 0.0);
  std::fill_n(Z_out->typed_data(), period * m * width, 0.0);
  for (int k = 0; k < period; ++k) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        T_out->typed_data()[(k * width + i) * width + j] =
            T_reordered[(k * n + i) * n + j];
      }
    }
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        Z_out->typed_data()[(k * m + i) * width + j] =
            Z_reordered[(k * m + i) * n + j];
      }
    }
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PeriodicSchurReorderRealF64,
    PeriodicSchurReorderRealImpl,
    ffi::Ffi::Bind()
        .Arg<F64R3>()
        .Arg<F64R3>()
        .Arg<PredR1>()
        .Arg<S32R0>()
        .Arg<F64R0>()
        .Ret<F64R3>()
        .Ret<F64R3>());

// Reorder an R-oriented complex periodic Schur form.
ffi::Error PeriodicSchurReorderComplexImpl(
    C128R3 T,
    C128R3 Z,
    PredR1 select,
    S32R0 schur_size,
    F64R0 tol,
    ffi::ResultBuffer<ffi::C128, 3> T_out,
    ffi::ResultBuffer<ffi::C128, 3> Z_out) {
  const int period = static_cast<int>(T.dimensions()[0]);
  const int width = static_cast<int>(T.dimensions()[1]);
  const int m = static_cast<int>(Z.dimensions()[1]);
  const int n = schur_size.typed_data()[0];
  if (T.dimensions()[2] != width ||
      Z.dimensions()[0] != period ||
      Z.dimensions()[2] != width ||
      select.dimensions()[0] != width ||
      n < 0 || n > width) {
    return ffi::Error::InvalidArgument(
        "T, Z, and select must share a static width containing schur_size");
  }
  const auto* selected =
      reinterpret_cast<const unsigned char*>(select.typed_data());
  using Complex = std::complex<double>;
  std::vector<Complex> T_live(period * n * n);
  std::vector<Complex> Z_live(period * m * n);
  std::vector<unsigned char> select_live(n);
  std::vector<Complex> T_reordered(period * n * n);
  std::vector<Complex> Z_reordered(period * m * n);
  for (int k = 0; k < period; ++k) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        T_live[(k * n + i) * n + j] =
            T.typed_data()[(k * width + i) * width + j];
      }
    }
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        Z_live[(k * m + i) * n + j] =
            Z.typed_data()[(k * m + i) * width + j];
      }
    }
  }
  for (int j = 0; j < n; ++j) {
    select_live[j] = selected[j];
  }
  const int info = compute_reordered_periodic_schur_Z(
      T_live.data(), Z_live.data(), select_live.data(),
      period, m, n, tol.typed_data()[0],
      T_reordered.data(), Z_reordered.data());
  if (info != 0) {
    return ffi::Error::Internal(
        "complex periodic Schur reordering failed with info=" +
        std::to_string(info));
  }
  std::fill_n(
      T_out->typed_data(), period * width * width, Complex(0.0, 0.0));
  std::fill_n(
      Z_out->typed_data(), period * m * width, Complex(0.0, 0.0));
  for (int k = 0; k < period; ++k) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        T_out->typed_data()[(k * width + i) * width + j] =
            T_reordered[(k * n + i) * n + j];
      }
    }
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        Z_out->typed_data()[(k * m + i) * width + j] =
            Z_reordered[(k * m + i) * n + j];
      }
    }
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PeriodicSchurReorderComplexC128,
    PeriodicSchurReorderComplexImpl,
    ffi::Ffi::Bind()
        .Arg<C128R3>()
        .Arg<C128R3>()
        .Arg<PredR1>()
        .Arg<S32R0>()
        .Arg<F64R0>()
        .Ret<C128R3>()
        .Ret<C128R3>());
