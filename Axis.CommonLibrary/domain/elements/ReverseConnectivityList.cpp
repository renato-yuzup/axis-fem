#include "ReverseConnectivityList.hpp"
#include "foundation/OutOfBoundsException.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/memory/pointer.hpp"

namespace ade = axis::domain::elements;
namespace afm = axis::foundation::memory;

ade::ReverseConnectivityList::ReverseConnectivityList(int maxEntryCount)
{
  count_ = maxEntryCount;
  afm::RelativePointer *array = &firstItem_;
  for (int i = 0; i < maxEntryCount; i++)
  {
    array[i] = afm::RelativePointer::NullPtr;
  }
}

ade::ReverseConnectivityList::~ReverseConnectivityList(void)
{
  count_ = 0;
}

void * ade::ReverseConnectivityList::operator new( size_t, void *ptr )
{
  return ptr;
}

void ade::ReverseConnectivityList::operator delete( void *, void * )
{
  // nothing to do here
}

int ade::ReverseConnectivityList::Count( void ) const
{
  return count_;
}

afm::RelativePointer ade::ReverseConnectivityList::GetItem( int index ) const
{
  if (index >= count_ || index < 0)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  const afm::RelativePointer *ptr = &firstItem_;
  return ptr[index];
}

void ade::ReverseConnectivityList::SetItem( int index, const afm::RelativePointer& ptr )
{
  if (index >= count_ || index < 0)
  {
    throw axis::foundation::OutOfBoundsException();
  }
  afm::RelativePointer *array = &firstItem_;
  array[index] = ptr;
}

afm::RelativePointer ade::ReverseConnectivityList::Create( int maxEntryCount )
{
  if (maxEntryCount < 0)
  {
    throw axis::foundation::ArgumentException();
  }
  int entryCount = maxEntryCount;
  if (maxEntryCount > 0) entryCount--;
  size_type memoryToAllocate = sizeof(ReverseConnectivityList) + entryCount*sizeof(afm::RelativePointer);
  afm::RelativePointer ptr = System::ModelMemory().Allocate(memoryToAllocate);
  new (*ptr) ReverseConnectivityList(maxEntryCount);
  return ptr;
}
