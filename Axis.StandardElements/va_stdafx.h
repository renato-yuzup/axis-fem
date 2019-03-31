/*
  This file is read by Visual AssistX in order to extend its parsing capabilities.
  The content below allows parsing CUDA files.

  Thanks to sunsetquest for sharing such valuable information!
  (original post: http://forums.wholetomato.com/forum/topic.asp?TOPIC_ID=10225)
*/

///////////////////////////////////////////////////////////// //////////////////
// macro for visual assist compatibility with cuda 
#define __launch_bounds__(x)
#define __restrict__ 
#define __device__
#define __global__
#define __shared__
#define __constant__
struct int3{
  int x;
  int y;
  int z;
};
struct uint3{
  unsigned int x;
  unsigned int y;
  unsigned int z;
};
struct blockIdx{
  int x;
  int y;
  int z;
};
struct threadIdx{
  unsigned int x;
  unsigned int y;
  unsigned int z;
};
struct blockDim{
  int x;
  int y;
  int z;
};
struct blockIdx{
  int x;
  int y;
  int z;
};
typedef int warpSize;
