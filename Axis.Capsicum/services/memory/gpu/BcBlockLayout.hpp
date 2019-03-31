#pragma once
#include "services/memory/MemoryLayout.hpp"

namespace axis { namespace services { namespace memory { namespace gpu {

/**
 * Describes memory layout of a boundary condition descriptor block.
 */
class BcBlockLayout : public axis::services::memory::MemoryLayout
{
public:

  /**
   * Constructor.
   *
   * @param specificDataSize Total length of data that is specific to boundary 
   *                         condition implementation.
   */
  BcBlockLayout(int specificDataSize);
  virtual ~BcBlockLayout(void);
  virtual MemoryLayout& Clone( void ) const;

  /**
   * Initialises the memory block.
   *
   * @param [in,out] targetBlock Base memory address of the target block.
   * @param dofId                Unique identifier of the associated degree of
   *                             freedom.
   */
  void InitMemoryBlock(void *targetBlock, uint64 dofId);

  /**
   * Returns the base address of boundary condition implementation-related data.
   *
   * @param [in,out] bcBaseAddress Base memory address of the boundary condition
   *                 descriptor block.
   *
   * @return The custom data base address.
   */
  void *GetCustomDataAddress(void *bcBaseAddress) const;

  /**
   * Returns the base address of boundary condition implementation-related data.
   *
   * @param [in,out] bcBaseAddress Base memory address of the boundary condition
   *                 descriptor block.
   *
   * @return The custom data base address.
   */
  const void *GetCustomDataAddress(const void *bcBaseAddress) const;

  /**
   * Returns the address of the output slot where the boundary condition shall
   * write its updated value.
   *
   * @param [in,out] bcBaseAddress Base memory address of the boundary condition
   *                 descriptor block.
   *
   * @return The output slot address.
   */
  real *GetOutputBucketAddress(void *bcBaseAddress) const;

  /**
   * Returns the address of the output slot where the boundary condition shall
   * write its updated value.
   *
   * @param [in,out] bcBaseAddress Base memory address of the boundary condition
   *                 descriptor block.
   *
   * @return The output slot address.
   */
  const real *GetOutputBucketAddress(const void *bcBaseAddress) const;

  /**
   * Returns the unique identifier of the degree of freedom associated to this
   * boundary condition.
   *
   * @param [in,out] bcBaseAddress Base memory address of the boundary condition
   *                 descriptor block.
   *
   * @return The degree of freedom identifier.
   */
  uint64 GetDofId(void *bcBaseAddress) const;
private:
  virtual size_type DoGetSegmentSize( void ) const;

  real blockSize_;
};

} } } } // namespace axis::services::memory::gpu
