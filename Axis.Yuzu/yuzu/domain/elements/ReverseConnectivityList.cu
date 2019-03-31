#include "ReverseConnectivityList.hpp"

namespace ayde = axis::yuzu::domain::elements;
namespace ayfm = axis::yuzu::foundation::memory;

GPU_ONLY ayde::ReverseConnectivityList::ReverseConnectivityList(int maxEntryCount)
{
  count_ = maxEntryCount;
  // private constructor; never called
}

GPU_ONLY ayde::ReverseConnectivityList::~ReverseConnectivityList(void)
{
  count_ = 0;
}

GPU_ONLY int ayde::ReverseConnectivityList::Count( void ) const
{
  return count_;
}

GPU_ONLY ayfm::RelativePointer ayde::ReverseConnectivityList::GetItem( int index ) const
{
  const ayfm::RelativePointer *ptr = &firstItem_;
  return ptr[index];
}
