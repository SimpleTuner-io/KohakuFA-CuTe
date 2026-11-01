// Standalone instruction/initialization diagnostic. No attention arithmetic.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string artifact_directory;
static unsigned module_count = 0;
static CUpti_SubscriberHandle subscriber;

static void capture(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id,
                    const void* data) {
  if (domain == CUPTI_CB_DOMAIN_RESOURCE && id == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
    const auto* resource = static_cast<const CUpti_ResourceData*>(data);
    const auto* module = static_cast<const CUpti_ModuleResourceData*>(resource->resourceDescriptor);
    const char needle[] = "tma4d_read";
    if (!module || !module->pCubin || !module->cubinSize) return;
    if (std::search(module->pCubin, module->pCubin + module->cubinSize,
                    needle, needle + sizeof(needle) - 1) == module->pCubin + module->cubinSize) return;
    const std::string path = artifact_directory + "/loaded-" + std::to_string(module_count++) + ".cubin";
    std::ofstream output(path, std::ios::binary);
    output.write(module->pCubin, module->cubinSize);
    std::ofstream events(artifact_directory + "/modules.jsonl", std::ios::app);
    events << "{\"module_id\":" << module->moduleId << ",\"bytes\":" << module->cubinSize
           << ",\"file\":\"loaded-" << module_count - 1 << ".cubin\"}\n";
  } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API || domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    const auto* info = static_cast<const CUpti_CallbackData*>(data);
    if (info->callbackSite != CUPTI_API_ENTER || !info->symbolName) return;
    if (!std::strstr(info->symbolName, "tma4d_")) return;
    std::ofstream events(artifact_directory + "/launches.jsonl", std::ios::app);
    events << "{\"symbol\":\"" << info->symbolName << "\",\"api\":\""
           << info->functionName << "\",\"context_uid\":" << info->contextUid << "}\n";
  }
}

static void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
static void check_driver(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    throw std::runtime_error(std::string("cuTensorMapEncodeTiled: ") + (name ? name : "unknown error"));
  }
}

// Elements are BF16 bit patterns, not floating-point arithmetic. All dimensions
// are nontrivial and every higher global stride includes a physical gap.
struct Layout {
  static constexpr int width = 64, heads = 2, batches = 2, tile_rows = 128;
  static constexpr int row_stride = 80, prefix = 64, guard = 64;
  int rows, head_stride, batch_stride, elements;
  explicit Layout(int n) : rows(n), head_stride((n + 3) * row_stride + 16),
                          batch_stride(3 * head_stride + 32),
                          elements(prefix + batches * batch_stride + guard) {}
  __host__ __device__ bool logical(int physical) const {
    int i = physical - prefix;
    if (i < 0) return false;
    const int batch = i / batch_stride;
    i %= batch_stride;
    const int head = i / head_stride;
    i %= head_stride;
    const int row = i / row_stride, column = i % row_stride;
    return batch < batches && head < heads && row < rows && column < width;
  }
};

constexpr uint16_t gap_bits = 0x5a5a;
constexpr uint16_t poison_bits = 0x7fc1;

extern "C" __global__ void tma4d_initialize(uint16_t* output, Layout layout, bool poison) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < layout.elements; i += gridDim.x * blockDim.x) {
    if (!layout.logical(i)) output[i] = gap_bits;
    else if (poison) output[i] = poison_bits;
  }
}

template <int mode>
__device__ __forceinline__ void produce(const CUtensorMap* map, uint16_t* output,
                                       Layout layout, const uint16_t* payload,
                                       uint16_t* shared) {
  const int y0 = blockIdx.x * Layout::tile_rows;
  const int head = blockIdx.y, batch = blockIdx.z;
  const uint16_t value = *payload;
  if constexpr (mode == 2) {
    for (int i = threadIdx.x; i < Layout::tile_rows * Layout::width; i += blockDim.x) {
      const int y = y0 + i / Layout::width, x = i % Layout::width;
      if (y < layout.rows) {
        output[Layout::prefix + batch * layout.batch_stride + head * layout.head_stride
               + y * Layout::row_stride + x] = value;
      }
    }
  } else {
    for (int i = threadIdx.x; i < Layout::tile_rows * Layout::width; i += blockDim.x) shared[i] = value;
    // Every writer publishes its own generic-proxy writes before the CTA joins.
    // This follows the CUDA Programming Guide TMA S2G example's ordering.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(shared));
      asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
                   "[%0, {%1, %2, %3, %4}], [%5];"
                   :: "l"(map), "r"(0), "r"(y0), "r"(head), "r"(batch), "r"(address) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      if constexpr (mode == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      else asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    // All source owners remain alive until the initiating thread's wait ends.
    __syncthreads();
  }
}

#define PRODUCER(name, mode) \
  extern "C" __global__ void name(const __grid_constant__ CUtensorMap map, uint16_t* output, \
                                  Layout layout, const uint16_t* payload) { \
    __shared__ __align__(128) uint16_t shared[Layout::tile_rows * Layout::width]; \
    produce<mode>(&map, output, layout, payload, shared); \
  }
PRODUCER(tma4d_read, 0)
PRODUCER(tma4d_full, 1)
PRODUCER(tma4d_ordinary, 2)
#undef PRODUCER

struct Verification { unsigned mismatches; unsigned first; };

extern "C" __global__ void tma4d_consume(const uint16_t* output, Layout layout,
                                         const uint16_t* payload, Verification* result) {
  // Read every physical 16-byte chunk, including gaps and both guards. The gap
  // initializer owns those bytes; it deliberately skips uninitialized logical O.
  for (int word = blockIdx.x * blockDim.x + threadIdx.x; word < layout.elements / 8;
       word += gridDim.x * blockDim.x) {
    uint32_t x, y, z, w;
    const uint16_t* address = output + word * 8;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address) : "memory");
    const uint32_t words[4] = {x, y, z, w};
    for (int lane = 0; lane < 8; ++lane) {
      const uint16_t value = static_cast<uint16_t>(words[lane / 2] >> ((lane % 2) * 16));
      const int physical = word * 8 + lane;
      const uint16_t expected = layout.logical(physical) ? *payload : gap_bits;
      if (value != expected) {
        atomicAdd(&result->mismatches, 1u);
        atomicMin(&result->first, static_cast<unsigned>(physical));
      }
    }
  }
}

static void launch(int mode, const CUtensorMap& map, uint16_t* output, Layout layout,
                   const uint16_t* payload, Verification* verification, cudaStream_t stream) {
  const dim3 grid((layout.rows + 127) / 128, Layout::heads, Layout::batches);
  if (mode == 0) tma4d_read<<<grid, 128, 0, stream>>>(map, output, layout, payload);
  else if (mode == 1) tma4d_full<<<grid, 128, 0, stream>>>(map, output, layout, payload);
  else tma4d_ordinary<<<grid, 128, 0, stream>>>(map, output, layout, payload);
  check(cudaGetLastError(), "producer launch");
  tma4d_consume<<<8, 128, 0, stream>>>(output, layout, payload, verification);
  check(cudaGetLastError(), "consumer launch");
}

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: tma4d_store read|full|ordinary uninitialized|poison 256|257 OUTPUT_DIR");
    const std::string mode_name = argv[1], initialization = argv[2];
    const int mode = mode_name == "read" ? 0 : mode_name == "full" ? 1 : mode_name == "ordinary" ? 2 : -1;
    const bool poison = initialization == "poison";
    const int rows = std::stoi(argv[3]);
    if (mode < 0 || (initialization != "poison" && initialization != "uninitialized") || (rows != 256 && rows != 257)) {
      throw std::runtime_error("unsupported diagnostic configuration");
    }
    artifact_directory = argv[4];  // Caller creates a fresh artifact directory.
    const char* capture_mode = std::getenv("TMA4D_CAPTURE");
    if (capture_mode && std::strcmp(capture_mode, "enabled") && std::strcmp(capture_mode, "disabled")) {
      throw std::runtime_error("TMA4D_CAPTURE must be explicitly enabled or disabled");
    }
    const bool capture_enabled = !capture_mode || !std::strcmp(capture_mode, "enabled");
    if (capture_enabled && (cuptiSubscribe(&subscriber, capture, nullptr) != CUPTI_SUCCESS ||
        cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED) != CUPTI_SUCCESS ||
        cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API) != CUPTI_SUCCESS ||
        cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API) != CUPTI_SUCCESS)) {
      throw std::runtime_error("CUPTI loaded-module/launch capture unavailable");
    }
    check(cudaSetDevice(0), "select device");
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0), "device properties");
    if (device.major < 9) throw std::runtime_error("generic TMA requires compute capability >= 9.0");
    int driver_version = 0, runtime_version = 0;
    check(cudaDriverGetVersion(&driver_version), "driver version");
    check(cudaRuntimeGetVersion(&runtime_version), "runtime version");
    cudaFuncAttributes attributes{};
    if (mode == 0) check(cudaFuncGetAttributes(&attributes, tma4d_read), "read producer attributes");
    else if (mode == 1) check(cudaFuncGetAttributes(&attributes, tma4d_full), "full producer attributes");
    else check(cudaFuncGetAttributes(&attributes, tma4d_ordinary), "ordinary producer attributes");
    Layout layout(rows);
    uint16_t *output = nullptr, *payload = nullptr;
    Verification* verification = nullptr;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check(cudaMalloc(&output, layout.elements * sizeof(uint16_t)), "allocate physical output");
    check(cudaMalloc(&payload, sizeof(uint16_t)), "allocate changing payload");
    check(cudaMalloc(&verification, sizeof(Verification)), "allocate verification");
    alignas(64) CUtensorMap map;
    const cuuint64_t dimensions[4] = {64, static_cast<cuuint64_t>(rows), 2, 2};
    const cuuint64_t strides[3] = {Layout::row_stride * 2,
                                  static_cast<cuuint64_t>(layout.head_stride) * 2,
                                  static_cast<cuuint64_t>(layout.batch_stride) * 2};
    const cuuint32_t box[4] = {64, 128, 1, 1}, element_strides[4] = {1, 1, 1, 1};
    check_driver(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                                      output + Layout::prefix, dimensions, strides, box, element_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    std::ofstream descriptor(artifact_directory + "/descriptor.bin", std::ios::binary);
    descriptor.write(reinterpret_cast<const char*>(&map), sizeof(map));
    descriptor.close();
    tma4d_initialize<<<8, 128, 0, stream>>>(output, layout, poison);
    check(cudaGetLastError(), "initialize gaps/optional poison");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::vector<uint16_t> physical(layout.elements);
    // Independent forward construction, not Layout.logical()'s inverse map.
    // A shared inverse-index bug must not hide missed logical writes or stores
    // into gaps. Enumerate each logical coordinate into an otherwise-gap mask.
    std::vector<uint8_t> host_logical(layout.elements, 0);
    for (int b = 0; b < 2; ++b) {
      for (int h = 0; h < 2; ++h) {
        for (int row = 0; row < rows; ++row) {
          for (int column = 0; column < 64; ++column) {
            const size_t offset = 64 + b * strides[2] / 2 + h * strides[1] / 2
                                  + row * strides[0] / 2 + column;
            if (offset >= host_logical.size() || host_logical[offset]) {
              throw std::runtime_error("independent host coordinate ownership is invalid");
            }
            host_logical[offset] = 1;
          }
        }
      }
    }
    std::ofstream report(artifact_directory + "/report.json");
    report << "{\"scope\":\"standalone generic TMA rank-4 store, no attention qualification\","
           << "\"capture_enabled\":" << (capture_enabled ? "true" : "false") << ","
           << "\"device\":\"" << device.name << "\",\"capability\":[" << device.major << "," << device.minor
           << "],\"driver\":" << driver_version << ",\"runtime\":" << runtime_version
           << ",\"sm_count\":" << device.multiProcessorCount
           << ",\"producer_resources\":{\"registers\":" << attributes.numRegs
           << ",\"static_shared_bytes\":" << attributes.sharedSizeBytes
           << ",\"local_bytes\":" << attributes.localSizeBytes
           << ",\"binary_version\":" << attributes.binaryVersion
           << ",\"ptx_version\":" << attributes.ptxVersion << ",\"block_threads\":128}"
           << ",\"mode\":\"" << mode_name << "\",\"initialization\":\"" << initialization
           << "\",\"dimensions\":[64," << rows << ",2,2],\"byte_strides\":["
           << strides[0] << "," << strides[1] << "," << strides[2] << "],\"box\":[64,128,1,1],"
           << "\"physical_allocation\":{\"ptr\":" << reinterpret_cast<uintptr_t>(output)
           << ",\"bytes\":" << layout.elements * 2 << ",\"logical_base_offset\":" << Layout::prefix * 2
           << "},\"iterations\":[";
    bool all_match = true;
    for (int generation = 0; generation < 3; ++generation) {
      const uint16_t value = static_cast<uint16_t>(0x3f80 + generation * 0x80); // BF16 1,2,4.
      const Verification initial{0, UINT32_MAX};
      if (poison && generation > 0) tma4d_initialize<<<8, 128, 0, stream>>>(output, layout, true);
      check(cudaMemcpyAsync(payload, &value, sizeof(value), cudaMemcpyHostToDevice, stream), "update payload");
      check(cudaMemcpyAsync(verification, &initial, sizeof(initial), cudaMemcpyHostToDevice, stream), "reset verification");
      if (generation == 0) {
        launch(mode, map, output, layout, payload, verification, stream);
      } else {
        if (!executable) {
          check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
          launch(mode, map, output, layout, payload, verification, stream);
          check(cudaStreamEndCapture(stream, &graph), "end graph capture");
          check(cudaGraphInstantiateWithFlags(&executable, graph, 0), "instantiate graph");
        }
        check(cudaGraphLaunch(executable, stream), "replay graph");
      }
      Verification result;
      check(cudaMemcpyAsync(&result, verification, sizeof(result), cudaMemcpyDeviceToHost, stream), "read verification");
      check(cudaMemcpyAsync(physical.data(), output, physical.size() * 2, cudaMemcpyDeviceToHost, stream), "read physical allocation");
      check(cudaStreamSynchronize(stream), "complete producer/consumer");
      unsigned cpu_mismatches = 0, first = UINT32_MAX;
      for (int i = 0; i < layout.elements; ++i) {
        const uint16_t expected = host_logical[i] ? value : gap_bits;
        if (physical[i] != expected) { ++cpu_mismatches; first = std::min(first, static_cast<unsigned>(i)); }
      }
      if (generation) report << ",";
      report << "{\"generation\":" << generation << ",\"bf16_bits\":" << value
             << ",\"graph\":" << (generation ? "true" : "false")
             << ",\"gpu_mismatches\":" << result.mismatches << ",\"gpu_first\":" << result.first
             << ",\"cpu_mismatches\":" << cpu_mismatches << ",\"cpu_first\":" << first << "}";
      report.flush();
      all_match &= !result.mismatches && !cpu_mismatches;
    }
    report << "],\"captured_module_count\":" << module_count << ",\"all_match\":"
           << (all_match ? "true" : "false") << "}\n";
    report.close();
    if (executable) check(cudaGraphExecDestroy(executable), "destroy graph executable");
    if (graph) check(cudaGraphDestroy(graph), "destroy graph");
    check(cudaFree(verification), "free verification");
    check(cudaFree(payload), "free payload");
    check(cudaFree(output), "free output");
    check(cudaStreamDestroy(stream), "destroy stream");
    if (capture_enabled) cuptiUnsubscribe(subscriber);
    return all_match && (!capture_enabled || module_count) ? 0 : 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
