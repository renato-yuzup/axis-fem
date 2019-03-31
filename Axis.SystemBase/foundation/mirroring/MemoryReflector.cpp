#include "MemoryReflector.hpp"
#include <vector>
#include "foundation/OutOfBoundsException.hpp"

namespace afmm = axis::foundation::mirroring;

class afmm::MemoryReflector::Pimpl
{
public:
  typedef std::pair<void *, uint64> block_info;
  typedef std::vector<block_info> block_list;
  block_list blocks;
};

afmm::MemoryReflector::MemoryReflector(void)
{
  pimpl_ = new Pimpl();
}

afmm::MemoryReflector::~MemoryReflector(void)
{
  delete pimpl_;
  pimpl_ = nullptr;
}

void afmm::MemoryReflector::AddBlock( void *blockStartingAddr, uint64 blockSize )
{
  pimpl_->blocks.push_back(std::make_pair(blockStartingAddr, blockSize));
}

int afmm::MemoryReflector::GetBlockCount( void ) const
{
  return pimpl_->blocks.size();
}

void * afmm::MemoryReflector::GetBlockStartAddress( int blockIndex ) const
{
  if (!(blockIndex >= 0 && blockIndex < GetBlockCount()))
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return pimpl_->blocks[blockIndex].first;
}

uint64 afmm::MemoryReflector::GetBlockSize( int blockIndex ) const
{
  if (!(blockIndex >= 0 && blockIndex < GetBlockCount()))
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return pimpl_->blocks[blockIndex].second;
}

void afmm::MemoryReflector::WriteToBlock( int destBlockIdx, const void *sourceAddress, uint64 dataSize )
{
  WriteToBlock(destBlockIdx, 0, sourceAddress, dataSize);
}

void afmm::MemoryReflector::Restore( void *destinationAddress, int srcBlockIdx, uint64 dataSize )
{
  Restore(destinationAddress, srcBlockIdx, 0, dataSize);
}
