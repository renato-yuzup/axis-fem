// #include "MultiLineCurve_GPU.hpp"
// #include "yuzu/foundation/memory/pointer.hpp"
// #include "domain/curves/MultiLineCurve.hpp"
// 
// #if defined(_DEBUG) || defined(DEBUG)
// #define cuda_assert(x) { if ((x) != CUDA_SUCCESS) throw 0xDEAD; }
// #else
// #define cuda_assert(x) x
// #endif
// 
// #define X(x)  points[2*(x) + 0]
// #define Y(x)  points[2*(x) + 1]
// 
// namespace adcu = axis::domain::curves;
// namespace ayfm = axis::yuzu::foundation::memory;
// 
// extern adcu::Curve::GPUCurveOperator_t adcu::__multiLineCurveOperatorAddr = nullptr;
// 
// __device__ real MultiLine_GPU_Curve_Op(void *curveDataPtr, real xCoord)
// {
//   ayfm::RelativePointer& dataPtr = *(ayfm::RelativePointer *)curveDataPtr;
//   void *curveDataRegion = *dataPtr;
//   uint64 numPoints = *(uint64 *)curveDataRegion;
//   const real *points = (real *)((uint64)curveDataRegion + sizeof(uint64));
//   for (size_t i = 1; i < numPoints; i++)
//   {
//     if ((X(i) > xCoord) || (i == numPoints-1 && (abs(X(i) - xCoord) <= 1e-15)))	
//     {	
//       // trivial case: horizontal line
// //       if (abs(Y(i-1) - Y(i)) <= 1e-15)
// //       {
// //         return Y(i);
// //       }
//       real a = Y(i)   * (xCoord - X(i-1));
//       real b = Y(i-1) * (xCoord - X(i));
//       real c = 1.0 / (X(i) - X(i-1));
//       a = a - b;
//       a = a * c;
//       return a;
// //       ret urn (a - b) / c;
//       // return (X(i)-X(i-1));
// //       return ((Y(i)-Y(i-1))) * (xCoord - X(i-1)) / (X(i)-X(i-1));
// //       return ((Y(i)-Y(i-1)) * (xCoord-X(i-1)) / (X(i)-X(i-1))) + Y(i-1);
//     }
//   }
//   return 0;
// }
// 
// __global__ void GetMultiLineCurveOperatorAddr(void *addr)
// {
//   *((adcu::Curve::GPUCurveOperator_t *)addr) = &MultiLine_GPU_Curve_Op;
// }
// 
// extern adcu::Curve::GPUCurveOperator_t 
//   adcu::GetMultiLineCurve_GPUOperatorAddress(void)
// {
//   if (__multiLineCurveOperatorAddr == nullptr)
//   {
//     void *devPtr;
//     cuda_assert(cudaMalloc(&devPtr, sizeof(void *)));
//     cudaStream_t stream;
//     cuda_assert(cudaStreamCreate(&stream));
//     GetMultiLineCurveOperatorAddr<<<1,1,0, stream>>>(devPtr);
//     cuda_assert(cudaStreamSynchronize(stream));
//     cuda_assert(cudaStreamDestroy(stream));
//     cuda_assert(cudaMemcpy(&__multiLineCurveOperatorAddr, devPtr, 
//       sizeof(void *), cudaMemcpyDeviceToHost));
//     cuda_assert(cudaFree(devPtr));
//   }
//   return __multiLineCurveOperatorAddr;
// }
