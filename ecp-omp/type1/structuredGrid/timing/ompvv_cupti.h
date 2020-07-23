#include "ompvv_cupti_string_cbid.h"

#define CUPTI_CALL(call)                                                       \
  do {                                                                         \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char *errstr;                                                      \
      cuptiGetResultString(_status, &errstr);                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",     \
              __FILE__, __LINE__, #call, errstr);                              \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#if defined(__xlc__) || defined(__xlC__)
/* IBM XL C/C++. -------------------------------------------- */
#define OMPVV_COMPILER_NAME "XLC"
#define OMPVV_COMPILER_VERSION __VERSION__

#elif defined(__clang__)
/* Clang/LLVM. ---------------------------------------------- */
#define OMPVV_COMPILER_NAME "clang"
#define OMPVV_COMPILER_VERSION __clang_version__

#elif defined(__GNUC__) || defined(__GNUG__)
/* GNU GCC/G++. --------------------------------------------- */
#define OMPVV_COMPILER_NAME "GCC"
#define OMPVV_COMPILER_VERSION __VERSION__

#endif

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static int ignoreEventsPrinting;
static int initOnlyOnlce = 0;
static uint64_t _ompvv_accum_driver, _ompvv_accum_kernel, _ompvv_accum_runtime,
    _ompvv_accum_memory, _ompvv_accum_others;

static const char *getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}

const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
  switch (kind) {
  case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
    return "COMPILER";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
    return "BUFFER_FLUSH";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
    return "INSTRUMENTATION";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
    return "RESOURCE";
  default:
    break;
  }

  return "<unknown>";
}

const char *getActivityObjectKindString(CUpti_ActivityObjectKind kind) {
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return "PROCESS";
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return "THREAD";
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return "DEVICE";
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return "CONTEXT";
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return "STREAM";
  default:
    break;
  }

  return "<unknown>";
}

uint32_t getActivityObjectKindId(CUpti_ActivityObjectKind kind,
                                 CUpti_ActivityObjectKindId *id) {
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

static const char *getComputeApiKindString(CUpti_ActivityComputeApiKind kind) {
  switch (kind) {
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
    return "CUDA";
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
    return "CUDA_MPS";
  default:
    break;
  }

  return "<unknown>";
}

static void printActivity(CUpti_Activity *record) {
  if (ignoreEventsPrinting)
    return;
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_DEVICE: {
    CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *)record;
    printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u "
           "GB/s, size %u MB), "
           "multiprocessors %u, clock %u MHz\n",
           device->name, device->id, device->computeCapabilityMajor,
           device->computeCapabilityMinor,
           (unsigned int)(device->globalMemoryBandwidth / 1024 / 1024),
           (unsigned int)(device->globalMemorySize / 1024 / 1024),
           device->numMultiprocessors,
           (unsigned int)(device->coreClockRate / 1000));
    break;
  }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: {
    CUpti_ActivityDeviceAttribute *attribute =
        (CUpti_ActivityDeviceAttribute *)record;
    printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
           attribute->attribute.cupti, attribute->deviceId,
           (unsigned long long)attribute->value.vUint64);
    break;
  }
  case CUPTI_ACTIVITY_KIND_CONTEXT: {
    CUpti_ActivityContext *context = (CUpti_ActivityContext *)record;
    printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
           context->contextId, context->deviceId,
           getComputeApiKindString(
               (CUpti_ActivityComputeApiKind)context->computeApiKind),
           (int)context->nullStreamId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;
    printf("MEMCPY \t %s \t %lu\n",
           getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
           (unsigned long)(memcpy->end - memcpy->start));
    _ompvv_accum_memory += (unsigned long long)(memcpy->end - memcpy->start);

    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMSET: {
    CUpti_ActivityMemset *memset = (CUpti_ActivityMemset *)record;
    printf("MEMSET \t %u \t %lu\n", memset->value,
           (unsigned long)(memset->end - memset->start));
    _ompvv_accum_memory += (unsigned long long)(memset->end - memset->start);
    break;
  }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    const char *kindString =
        (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
    CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
    printf("%s \t \"%s\"    GPU %u    %lu\n", kindString, kernel->name, kernel->deviceId,
           (unsigned long)(kernel->end - kernel->start));
    _ompvv_accum_kernel += (unsigned long long)(kernel->end - kernel->start);
    break;
  }
  case CUPTI_ACTIVITY_KIND_DRIVER: {
    CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
    printf("DRIVER \t %s %lu\n",
           getDriverApiKindString(api->cbid),
           (unsigned long)(api->end - api->start));
    _ompvv_accum_driver += (unsigned long long)(api->end - api->start);
    break;
  }
  case CUPTI_ACTIVITY_KIND_RUNTIME: {
    CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
    printf("RUNTIME \t %s \t %lu\n",
           getRuntimeApiKindString(api->cbid),
           (unsigned long)(api->end - api->start));
    _ompvv_accum_runtime += (unsigned long long)(api->end - api->start);
    break;
  }
  case CUPTI_ACTIVITY_KIND_NAME: {
    CUpti_ActivityName *name = (CUpti_ActivityName *)record;
    switch (name->objectKind) {
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      printf("NAME  %s %u %s id %u, name %s\n",
             getActivityObjectKindString(name->objectKind),
             getActivityObjectKindId(name->objectKind, &name->objectId),
             getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
             getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE,
                                     &name->objectId),
             name->name);
      break;
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      printf("NAME %s %u %s %u %s id %u, name %s\n",
             getActivityObjectKindString(name->objectKind),
             getActivityObjectKindId(name->objectKind, &name->objectId),
             getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
             getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT,
                                     &name->objectId),
             getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
             getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE,
                                     &name->objectId),
             name->name);
      break;
    default:
      printf("NAME %s id %u, name %s\n",
             getActivityObjectKindString(name->objectKind),
             getActivityObjectKindId(name->objectKind, &name->objectId),
             name->name);
      break;
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_MARKER: {
    CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *)record;
    printf("MARKER id %u [ %llu ], name %s, domain %s\n", marker->id,
           (unsigned long long)marker->timestamp, marker->name, marker->domain);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MARKER_DATA: {
    CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *)record;
    printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
           marker->id, marker->color, marker->category,
           (unsigned long long)marker->payload.metricValueUint64,
           marker->payload.metricValueDouble);
    break;
  }
  case CUPTI_ACTIVITY_KIND_OVERHEAD: {
    CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *)record;
    printf("OVERHEAD \t %s \t %llu\n",
           getActivityOverheadKindString(overhead->overheadKind),
           (unsigned long long)(overhead->end - overhead->start));
    _ompvv_accum_others +=
        (unsigned long long)(overhead->end - overhead->start);
    break;
  }
  default:
    printf("  <unknown>\n");
    break;
  }
  fflush(stdout);
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
  uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int)dropped);
    }
  }

  fflush(stdout);
  free(buffer);
}

void initTrace() {
  if (initOnlyOnlce)
    return;
  initOnlyOnlce = 1;
  size_t attrValue = 0, attrValueSize = sizeof(size_t);
  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  // Enable all other activity record kinds.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));

  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // Get and set activity attributes.
  // Attributes can be set by the CUPTI client to change behavior of the
  // activity API. Some attributes require to be set before any CUDA context is
  // created to be effective, e.g. to be applied to all device buffer
  // allocations (see documentation).
  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                       &attrValueSize, &attrValue));
  //  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long
  //  unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                       &attrValueSize, &attrValue));

  CUPTI_CALL(
      cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                &attrValueSize, &attrValue));
  //  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT",
  //  (long long unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(
      cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                &attrValueSize, &attrValue));

  ignoreEventsPrinting = 1;
}
