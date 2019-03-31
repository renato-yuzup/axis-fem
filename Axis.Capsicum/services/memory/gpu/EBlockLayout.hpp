#pragma once
#include "services/memory/MemoryLayout.hpp"

namespace axis { namespace services { namespace memory { namespace gpu {

/**
 * Describes memory layout of a finite element descriptor block.
 */
class EBlockLayout : public axis::services::memory::MemoryLayout
{
public:

  /**
   * Constructor.
   *
   * @param formulationBlockSize Size of the formulation data block.
   * @param materialBlockSize    Size of the material data block.
   */
  EBlockLayout(size_type formulationBlockSize, size_type materialBlockSize);
  ~EBlockLayout(void);
  virtual MemoryLayout& Clone( void ) const;

  /**
   * Returns the formulation data block address.
   *
   * @param [in,out] elementBaseAddress Base address of the finite element 
   *                 descriptor block.
   *
   * @return The formulation data block base address.
   */
  void *GetFormulationBlockAddress(void *elementBaseAddress) const;

  /**
   * Returns the material data block address.
   *
   * @param [in,out] elementBaseAddress Base address of the finite element 
   *                 descriptor block.
   *
   * @return The material data block base address.
   */
  void *GetMaterialBlockAddress(void *elementBaseAddress) const;

  /**
   * Returns the identifier of the element associated to a block.
   *
   * @param [in,out] elementBaseAddress Base address of the finite element 
   *                 descriptor block.
   *
   * @return The element identifier.
   */
  uint64 GetElementId(void *elementBaseAddress) const;

  /**
   * Initialises the element memory block.
   *
   * @param [in,out] elementBaseAddress Base address of the finite element 
   *                 descriptor block.
   * @param elementId                   Identifier of the associated element.
   * @param totalDofCount               Total number of degrees of freedom.
   */
  void InitMemoryBlock(void *elementBaseAddress, uint64 elementId, 
    int totalDofCount);
private:
  virtual size_type DoGetSegmentSize( void ) const;
  size_type formulationSize_;
  size_type materialSize_;
};

} } } } // namespace axis::services::memory::gpu
