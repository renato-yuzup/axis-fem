#pragma once
#include "services/memory/MemoryLayout.hpp"

namespace axis { namespace services { namespace memory { namespace gpu {

/**
 * Describes memory layout of a curve descriptor block.
 */
class CBlockLayout : public axis::services::memory::MemoryLayout
{
public:
  /**
   * Constructor.
   *
   * @param specificDataSize Total length of data that is specific to curve
   *                         implementation.
   */
  CBlockLayout(int specificDataSize);
  virtual ~CBlockLayout(void);
  virtual MemoryLayout& Clone( void ) const;

  /**
   * Initialises the memory block.
   *
   * @param [in,out] targetBlock Base memory address of the target block.
   */
  void InitMemoryBlock(void *targetBlock);

  /**
   * Returns the base address of curve implementation-related data.
   *
   * @param [in,out] bcBaseAddress Base memory address of the curve descriptor 
   *                 block.
   *
   * @return The custom data base address.
   */
  void *GetCustomDataAddress(void *curveBaseAddress) const;

  /**
   * Returns the base address of curve implementation-related data.
   *
   * @param [in,out] bcBaseAddress Base memory address of the curve descriptor 
   *                 block.
   *
   * @return The custom data base address.
   */
  const void *GetCustomDataAddress(const void *curveBaseAddress) const;

  /**
   * Returns the address of the output slot where the curve shall write its 
   * updated value.
   *
   * @param [in,out] bcBaseAddress Base memory address of the curve descriptor 
   *                 block.
   *
   * @return The output slot address.
   */
  real *GetOutputBucketAddress(void *curveBaseAddress) const;

  /**
   * Returns the address of the output slot where the curve shall write its 
   * updated value.
   *
   * @param [in,out] bcBaseAddress Base memory address of the curve descriptor 
   *                 block.
   *
   * @return The output slot address.
   */
  const real *GetOutputBucketAddress(const void *curveBaseAddress) const;
private:
  virtual size_type DoGetSegmentSize( void ) const;
  int blockSize_;
};

} } } } // namespace axis::services::memory::gpu
