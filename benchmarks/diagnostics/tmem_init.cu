// Standalone TMEM allocator initialization diagnostic; no attention or tensor operations.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string output_directory;
static unsigned module_count = 0;
static bool capture_failed = false;
static CUpti_SubscriberHandle subscriber;

static void capture(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
  if (domain == CUPTI_CB_DOMAIN_RESOURCE && id == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
    const auto* resource = static_cast<const CUpti_ResourceData*>(data);
    const auto* module = static_cast<const CUpti_ModuleResourceData*>(resource->resourceDescriptor);
    const char needle[] = "tmem_init_allocation";
    if (!module || !module->pCubin || !module->cubinSize) return;
    if (std::search(module->pCubin, module->pCubin + module->cubinSize,
                    needle, needle + sizeof(needle) - 1) == module->pCubin + module->cubinSize) return;
    const std::string file = "loaded-" + std::to_string(module_count++) + ".cubin";
    std::ofstream binary(output_directory + "/" + file, std::ios::binary);
    binary.write(module->pCubin, module->cubinSize);
    if (!binary) capture_failed = true;
    std::ofstream events(output_directory + "/modules.jsonl", std::ios::app);
    events << "{\"module_id\":" << module->moduleId << ",\"bytes\":" << module->cubinSize
           << ",\"file\":\"" << file << "\"}\n";
    if (!events) capture_failed = true;
  } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API || domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    const auto* info = static_cast<const CUpti_CallbackData*>(data);
    if (info->callbackSite != CUPTI_API_ENTER || !info->symbolName || !std::strstr(info->symbolName, "tmem_init_")) return;
    std::ofstream events(output_directory + "/launches.jsonl", std::ios::app);
    events << "{\"symbol\":\"" << info->symbolName << "\",\"api\":\"" << info->functionName
           << "\",\"context_uid\":" << info->contextUid << "}\n";
    if (!events) capture_failed = true;
  }
}

static void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Observation { uint32_t pointer, generation; };
constexpr uint32_t untouched = 0xa5a5a5a5;
constexpr uint32_t sentinel = 0x7fffffff;

__device__ __forceinline__ void role_join(bool roles) {
  if (roles) asm volatile("barrier.sync 2, 416;" ::: "memory");
  else __syncthreads();
}

template <bool allocation>
__device__ __forceinline__ void inspect(Observation* output, const uint32_t* generation,
                                       volatile uint32_t* pointer, bool poison, bool roles) {
  const unsigned tid = threadIdx.x, warp = tid / 32;
  if (poison && tid == 0) *pointer = sentinel;
  __syncthreads();
  // Match forward: softmax warps 0..7, correction 8..11 and MMA warp 12.
  // Warps 13..15 do not participate in the role barrier or read the pointer.
  const bool participating = !roles || warp <= 12;
  if (participating) {
    if (warp == 12) {
      if constexpr (allocation) {
        const uint32_t shared_address = static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<uint32_t*>(pointer)));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                     :: "r"(shared_address) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
      } else if (tid == 384) {
        *pointer = 0;
      }
    }
    role_join(roles);
    const uint32_t actual = *pointer;
    output[tid] = Observation{actual, *generation};
    role_join(roles);
    if constexpr (allocation) {
      if (warp == 12) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(actual) : "memory");
      }
    }
  }
}

extern "C" __global__ void tmem_init_allocation(Observation* output, const uint32_t* generation, bool poison, bool roles) {
  __shared__ uint32_t pointer;
  inspect<true>(output, generation, &pointer, poison, roles);
}

extern "C" __global__ void tmem_init_ordinary(Observation* output, const uint32_t* generation, bool poison, bool roles) {
  __shared__ uint32_t pointer;
  inspect<false>(output, generation, &pointer, poison, roles);
}

static void launch(bool allocation, Observation* output, const uint32_t* generation, bool poison,
                   bool roles, cudaStream_t stream) {
  if (allocation) tmem_init_allocation<<<1, 512, 0, stream>>>(output, generation, poison, roles);
  else tmem_init_ordinary<<<1, 512, 0, stream>>>(output, generation, poison, roles);
  check(cudaGetLastError(), "diagnostic launch");
}

int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::runtime_error("Expected producer, initialization, barrier, capture, output directory.");
    const std::string producer = argv[1], initialization = argv[2], barrier = argv[3], capture_mode = argv[4];
    if (producer != "allocation" && producer != "ordinary") throw std::runtime_error("Unknown producer.");
    if (initialization != "untouched" && initialization != "sentinel") throw std::runtime_error("Unknown initialization.");
    if (barrier != "cta" && barrier != "roles") throw std::runtime_error("Unknown barrier.");
    if (capture_mode != "enabled" && capture_mode != "disabled") throw std::runtime_error("Unknown capture mode.");
    output_directory = argv[5];
    if (!std::filesystem::is_directory(output_directory) ||
        std::filesystem::directory_iterator(output_directory) != std::filesystem::directory_iterator())
      throw std::runtime_error("The output directory must exist and be empty; preserve completed evidence.");
    const bool allocation = producer == "allocation", poison = initialization == "sentinel";
    const bool roles = barrier == "roles", capture_enabled = capture_mode == "enabled";
    if (capture_enabled) {
      if (cuptiSubscribe(&subscriber, capture, nullptr) != CUPTI_SUCCESS ||
          cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE) != CUPTI_SUCCESS ||
          cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API) != CUPTI_SUCCESS ||
          cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API) != CUPTI_SUCCESS)
        throw std::runtime_error("Required loaded-code capture unavailable.");
    }
    check(cudaSetDevice(0), "select device");
    cudaDeviceProp device;
    check(cudaGetDeviceProperties(&device, 0), "device properties");
    if (device.major != 10 || (device.minor != 0 && device.minor != 3))
      throw std::runtime_error("This diagnostic requires datacentre Blackwell.");
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver), "driver version");
    check(cudaRuntimeGetVersion(&runtime), "runtime version");
    cudaFuncAttributes attributes;
    if (allocation) check(cudaFuncGetAttributes(&attributes, tmem_init_allocation), "allocation resources");
    else check(cudaFuncGetAttributes(&attributes, tmem_init_ordinary), "ordinary resources");
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "create stream");
    Observation* output;
    uint32_t* generation;
    check(cudaMalloc(&output, 512 * sizeof(Observation)), "allocate observations");
    check(cudaMalloc(&generation, sizeof(uint32_t)), "allocate generation");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::vector<Observation> observed(512);
    std::ofstream report(output_directory + "/report.json");
    if (!report) throw std::runtime_error("Cannot create diagnostic report.");
    report << "{\"scope\":\"standalone TMEM allocation; no attention qualification\","
           << "\"capture_enabled\":" << (capture_enabled ? "true" : "false")
           << ",\"device\":\"" << device.name << "\",\"capability\":[" << device.major << "," << device.minor
           << "],\"sm_count\":" << device.multiProcessorCount << ",\"driver\":" << driver << ",\"runtime\":" << runtime
           << ",\"producer\":\"" << producer << "\",\"initialization\":\"" << initialization
           << "\",\"barrier\":\"" << barrier << "\",\"tmem_columns\":512,\"block_threads\":512,\"resources\":{"
           << "\"registers\":" << attributes.numRegs << ",\"shared_bytes\":" << attributes.sharedSizeBytes
           << ",\"local_bytes\":" << attributes.localSizeBytes << "},\"iterations\":[";
    bool all_match = true;
    for (uint32_t current = 1; current <= 3; ++current) {
      check(cudaMemsetAsync(output, 0xa5, observed.size() * sizeof(Observation), stream), "reset output guards");
      check(cudaMemcpyAsync(generation, &current, sizeof(current), cudaMemcpyHostToDevice, stream), "change generation");
      if (current == 1) launch(allocation, output, generation, poison, roles, stream);
      else {
        if (!executable) {
          check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
          launch(allocation, output, generation, poison, roles, stream);
          check(cudaStreamEndCapture(stream, &graph), "end capture");
          check(cudaGraphInstantiateWithFlags(&executable, graph, 0), "instantiate graph");
        }
        check(cudaGraphLaunch(executable, stream), "replay graph");
      }
      check(cudaMemcpyAsync(observed.data(), output, observed.size() * sizeof(Observation), cudaMemcpyDeviceToHost, stream),
            "read observations");
      check(cudaStreamSynchronize(stream), "complete diagnostic");
      unsigned mismatches = 0, participants = 0;
      for (unsigned tid = 0; tid < observed.size(); ++tid) {
        const bool participating = !roles || tid < 416;
        const uint32_t expected_pointer = participating ? 0 : untouched;
        const uint32_t expected_generation = participating ? current : untouched;
        participants += participating;
        mismatches += observed[tid].pointer != expected_pointer || observed[tid].generation != expected_generation;
      }
      if (current != 1) report << ",";
      report << "{\"generation\":" << current << ",\"graph\":" << (current == 1 ? "false" : "true")
             << ",\"participants\":" << participants << ",\"mismatches\":" << mismatches << "}";
      report.flush();
      all_match &= mismatches == 0;
    }
    report << "],\"captured_modules\":" << module_count << ",\"all_match\":" << (all_match ? "true" : "false") << "}\n";
    report.close();
    if (executable) check(cudaGraphExecDestroy(executable), "destroy graph executable");
    if (graph) check(cudaGraphDestroy(graph), "destroy graph");
    check(cudaFree(generation), "free generation");
    check(cudaFree(output), "free observations");
    check(cudaStreamDestroy(stream), "destroy stream");
    if (capture_enabled) cuptiUnsubscribe(subscriber);
    return all_match && !capture_failed && (!capture_enabled || module_count) ? 0 : 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
