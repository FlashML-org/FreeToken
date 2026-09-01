// Minimal thrust::complex stand-in for the HIP JIT build: TheRock ROCm wheels
// ship no rocThrust, but torch's c10/util/complex{,_math}.h hard-include
// <thrust/complex.h> and call thrust::log/exp/pow/... under __HIPCC__. This
// header provides exactly that surface with the standard identities; the GGUF
// kernels themselves never execute complex math, but the functions must parse
// and link for float/double.
#pragma once
#include <cmath>

#define FT_THRUST_HD __host__ __device__

namespace thrust {

template <typename T>
class complex {
  T re_, im_;

 public:
  FT_THRUST_HD complex(T re = T(), T im = T()) : re_(re), im_(im) {}
  template <typename U>
  FT_THRUST_HD complex(const complex<U>& o)
      : re_(static_cast<T>(o.real())), im_(static_cast<T>(o.imag())) {}
  FT_THRUST_HD T real() const { return re_; }
  FT_THRUST_HD T imag() const { return im_; }
};

template <typename T>
FT_THRUST_HD inline complex<T> operator+(const complex<T>& a, const complex<T>& b) {
  return complex<T>(a.real() + b.real(), a.imag() + b.imag());
}
template <typename T>
FT_THRUST_HD inline complex<T> operator-(const complex<T>& a, const complex<T>& b) {
  return complex<T>(a.real() - b.real(), a.imag() - b.imag());
}
template <typename T>
FT_THRUST_HD inline complex<T> operator-(const complex<T>& a) {
  return complex<T>(-a.real(), -a.imag());
}
template <typename T>
FT_THRUST_HD inline complex<T> operator*(const complex<T>& a, const complex<T>& b) {
  return complex<T>(a.real() * b.real() - a.imag() * b.imag(),
                    a.real() * b.imag() + a.imag() * b.real());
}
template <typename T>
FT_THRUST_HD inline complex<T> operator*(const complex<T>& a, const T& s) {
  return complex<T>(a.real() * s, a.imag() * s);
}
template <typename T>
FT_THRUST_HD inline complex<T> operator*(const T& s, const complex<T>& a) {
  return a * s;
}
template <typename T>
FT_THRUST_HD inline complex<T> operator/(const complex<T>& a, const complex<T>& b) {
  T d = b.real() * b.real() + b.imag() * b.imag();
  return complex<T>((a.real() * b.real() + a.imag() * b.imag()) / d,
                    (a.imag() * b.real() - a.real() * b.imag()) / d);
}

template <typename T>
FT_THRUST_HD inline T abs(const complex<T>& z) {
  return ::hypot(z.real(), z.imag());
}
template <typename T>
FT_THRUST_HD inline T arg(const complex<T>& z) {
  return ::atan2(z.imag(), z.real());
}
template <typename T>
FT_THRUST_HD inline complex<T> polar(const T& r, const T& theta = T()) {
  return complex<T>(r * ::cos(theta), r * ::sin(theta));
}
template <typename T>
FT_THRUST_HD inline complex<T> proj(const complex<T>& z) {
  return z;
}

template <typename T>
FT_THRUST_HD inline complex<T> exp(const complex<T>& z) {
  return polar<T>(::exp(z.real()), z.imag());
}
template <typename T>
FT_THRUST_HD inline complex<T> log(const complex<T>& z) {
  return complex<T>(::log(abs(z)), arg(z));
}
template <typename T>
FT_THRUST_HD inline complex<T> log10(const complex<T>& z) {
  return log(z) * static_cast<T>(0.43429448190325182765);  // 1/ln(10)
}
template <typename T>
FT_THRUST_HD inline complex<T> sqrt(const complex<T>& z) {
  return polar<T>(::sqrt(abs(z)), arg(z) / T(2));
}
template <typename T>
FT_THRUST_HD inline complex<T> pow(const complex<T>& a, const complex<T>& b) {
  return exp(b * log(a));
}
template <typename T>
FT_THRUST_HD inline complex<T> pow(const complex<T>& a, const T& b) {
  return exp(log(a) * b);
}
template <typename T>
FT_THRUST_HD inline complex<T> pow(const T& a, const complex<T>& b) {
  return exp(b * log(complex<T>(a)));
}

template <typename T>
FT_THRUST_HD inline complex<T> sin(const complex<T>& z) {
  return complex<T>(::sin(z.real()) * ::cosh(z.imag()),
                    ::cos(z.real()) * ::sinh(z.imag()));
}
template <typename T>
FT_THRUST_HD inline complex<T> cos(const complex<T>& z) {
  return complex<T>(::cos(z.real()) * ::cosh(z.imag()),
                    -::sin(z.real()) * ::sinh(z.imag()));
}
template <typename T>
FT_THRUST_HD inline complex<T> tan(const complex<T>& z) {
  return sin(z) / cos(z);
}
template <typename T>
FT_THRUST_HD inline complex<T> sinh(const complex<T>& z) {
  return complex<T>(::sinh(z.real()) * ::cos(z.imag()),
                    ::cosh(z.real()) * ::sin(z.imag()));
}
template <typename T>
FT_THRUST_HD inline complex<T> cosh(const complex<T>& z) {
  return complex<T>(::cosh(z.real()) * ::cos(z.imag()),
                    ::sinh(z.real()) * ::sin(z.imag()));
}
template <typename T>
FT_THRUST_HD inline complex<T> tanh(const complex<T>& z) {
  return sinh(z) / cosh(z);
}

template <typename T>
FT_THRUST_HD inline complex<T> asinh(const complex<T>& z) {
  return log(z + sqrt(z * z + complex<T>(T(1))));
}
template <typename T>
FT_THRUST_HD inline complex<T> acosh(const complex<T>& z) {
  return log(z + sqrt(z * z - complex<T>(T(1))));
}
template <typename T>
FT_THRUST_HD inline complex<T> atanh(const complex<T>& z) {
  const complex<T> one(T(1));
  return log((one + z) / (one - z)) * T(0.5);
}
template <typename T>
FT_THRUST_HD inline complex<T> asin(const complex<T>& z) {
  const complex<T> i(T(0), T(1));
  const complex<T> w = asinh(i * z);
  return complex<T>(w.imag(), -w.real());  // -i * asinh(i z)
}
template <typename T>
FT_THRUST_HD inline complex<T> acos(const complex<T>& z) {
  const complex<T> w = asin(z);
  return complex<T>(static_cast<T>(1.57079632679489661923) - w.real(), -w.imag());
}
template <typename T>
FT_THRUST_HD inline complex<T> atan(const complex<T>& z) {
  const complex<T> i(T(0), T(1));
  const complex<T> w = atanh(i * z);
  return complex<T>(w.imag(), -w.real());  // -i * atanh(i z)
}

}  // namespace thrust

#undef FT_THRUST_HD
