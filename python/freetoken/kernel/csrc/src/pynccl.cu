#include <freetoken/nccl227.h>
#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

using NCCLIDList = tvm::ffi::Array<char>;

auto NCCL_CHECK(::ncclResult_t result) -> void {
  if (result != ::ncclSuccess) {
    host::RuntimeCheck(false, ::ncclGetErrorString(result));
  }
}

auto get_uid(const NCCLIDList &wrapper) -> ncclUniqueId {
  host::RuntimeCheck(wrapper.size() == NCCL_UNIQUE_ID_BYTES,
                     "Invalid NCCL ID wrapper size");
  ncclUniqueId id;
  std::copy(wrapper.begin(), wrapper.end(), id.internal);
  return id;
}

auto create_uid() -> NCCLIDList {
  ncclUniqueId id;
  NCCL_CHECK(::ncclGetUniqueId(&id));
  return NCCLIDList(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

const auto kNCCLReduceOPMap = std::unordered_map<std::string_view, ncclRedOp_t>{
    {"sum", ::ncclSum}, {"prod", ::ncclProd}, {"max", ::ncclMax},
    {"min", ::ncclMin}, {"avg", ::ncclAvg},
};

struct DLDataTypeHash {
  auto operator()(const DLDataType &dtype) const noexcept -> std::size_t {
    return std::bit_cast<std::uint32_t>(dtype);
  }
};

template <typename = void>
auto operator==(const DLDataType &a, const DLDataType &b) -> bool {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

const auto kNCCLDtypeMap =
    std::unordered_map<DLDataType, ncclDataType_t, DLDataTypeHash>{
        {{DLDataTypeCode::kDLFloat, 16, 1}, ncclFloat16},
        {{DLDataTypeCode::kDLBfloat, 16, 1}, ncclBfloat16},
    };

using std::shared_ptr;

template <typename T> using shared_obj = shared_ptr<std::remove_pointer_t<T>>;
template <auto Fn>
inline constexpr auto template_fn =
    [](auto &&...args) { return Fn(std::forward<decltype(args)>(args)...); };

struct NCCLWrapper : public tvm::ffi::Object {
public:
  NCCLWrapper(int rank, int world_size, const size_t max_bytes, NCCLIDList uid)
      : m_rank(rank), m_world_size(world_size) {
    (void)max_bytes;
    ncclUniqueId id = get_uid(uid);
    ncclComm_t comm;
    NCCL_CHECK(::ncclCommInitRank(&comm, m_world_size, id, m_rank));
    m_comm = {comm, template_fn<::ncclCommDestroy>};
  }

  auto all_reduce(tvm::ffi::TensorView t, std::string op) const -> void {
    using namespace host;
    RuntimeCheck(t.device().device_type == kDLCUDA,
                 "Tensor must be on CUDA device");
    RuntimeCheck(t.is_contiguous(), "Tensor must be contiguous");
    const auto size_dim = static_cast<size_t>(t.shape().Product());
    const auto dtype = kNCCLDtypeMap.at(t.dtype());
    const auto data_ptr = t.data_ptr();
    const auto reduce_op = kNCCLReduceOPMap.at(op);
    const auto stream = LaunchKernel::resolve_device(t.device());
    NCCL_CHECK(::ncclAllReduce(
        /*sendbuff=*/data_ptr,
        /*recvbuff=*/data_ptr,
        /*count=*/size_dim,
        /*datatype=*/dtype,
        /*op=*/reduce_op,
        /*comm=*/m_comm.get(),
        /*stream=*/stream));
  }

  auto all_gather(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) const
      -> void {
    using namespace host;
    RuntimeCheck(src.device().device_type == kDLCUDA,
                 "Tensor must be on CUDA device");
    RuntimeCheck(src.is_contiguous(), "Tensor must be contiguous");
    RuntimeCheck(dst.device().device_type == kDLCUDA,
                 "Tensor must be on CUDA device");
    RuntimeCheck(dst.is_contiguous(), "Tensor must be contiguous");
    RuntimeCheck(dst.size(0) == src.size(0) * m_world_size,
                 "Destination tensor has incorrect size");
    const auto size_dim = static_cast<size_t>(src.shape().Product());
    const auto dtype = kNCCLDtypeMap.at(src.dtype());
    const auto src_ptr = src.data_ptr();
    const auto dst_ptr = dst.data_ptr();
    const auto stream = LaunchKernel::resolve_device(src.device());
    // do not use internal buffer for all_gather, directly gather to output
    // tensor
    NCCL_CHECK(::ncclAllGather(
        /*sendbuff=*/src_ptr,
        /*recvbuff=*/dst_ptr,
        /*sendcount=*/size_dim,
        /*datatype=*/dtype,
        /*comm=*/m_comm.get(),
        /*stream=*/stream));
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("freetoken.NCCLWrapper", NCCLWrapper,
                                    tvm::ffi::Object);

private:
  int m_rank;
  int m_world_size;
  shared_obj<ncclComm_t> m_comm;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<NCCLWrapper>()
      .def(refl::init<int, int, size_t, NCCLIDList>(), "__init__")
      .def("all_reduce", &NCCLWrapper::all_reduce)
      .def("all_gather", &NCCLWrapper::all_gather);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(create_nccl_uid, &create_uid);

} // namespace
