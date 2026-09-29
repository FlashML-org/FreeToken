#include <freetoken/tensor.h>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

auto test(tvm::ffi::TensorView x, tvm::ffi::TensorView y) -> void {
  auto N = host::SymbolicSize{"N"};
  const auto M = 1024;
  host::TensorMatcher({N, M})
      .with_strides({-1, 1}) // -1 means any
      .template with_dtype<int, float>()
      .template with_device<kDLCPU>()
      .verify(x);
  host::TensorMatcher({N, M}) // default contiguous
      .with_dtype({{kDLInt, 32, 1}, {kDLInt, 64, 1}})
      .with_device({{kDLCUDA, 1}})
      .verify(y);
  host::RuntimeCheck(N.unwrap() % 4 == 0);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(test, test);
