#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"

namespace axis { namespace yuzu { namespace domain { namespace elements {

class ReverseConnectivityList
{
public:
  GPU_ONLY ~ReverseConnectivityList(void);  
  GPU_ONLY int Count(void) const;
  GPU_ONLY axis::yuzu::foundation::memory::RelativePointer GetItem(int index) const;
private:
  GPU_ONLY ReverseConnectivityList(int connectivityCount);

  int count_;
  axis::yuzu::foundation::memory::RelativePointer firstItem_;

  ReverseConnectivityList(const ReverseConnectivityList& );
  ReverseConnectivityList& operator =(const ReverseConnectivityList& );
};

} } } } // namespace axis::yuzu::domain::elements
