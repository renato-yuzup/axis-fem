#pragma once
#include "MemoryCommand.hpp"
#include "services/memory/MemoryLayout.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace services { namespace memory { namespace commands {

/**
 * Represents a command that operates on several memory blocks with the same
 * structure and data semantics.
 */
class ElementInitializerCommand : public MemoryCommand
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] associatedDataPtr Array of pointers pointing to the
   *                 data associated with each memory block (in the same order
   *                 as the memory blocks).
   * @param elementCount               Number of entities (blocks).
   */
  ElementInitializerCommand(void **associatedDataPtr, 
                            uint64 elementCount);
  virtual ~ElementInitializerCommand(void);
  MemoryCommand& Clone( void ) const;
  virtual void Execute( void *cpuBaseAddress, void *gpuBaseAddress );

  /**
   * Returns the entity count (equals the number of blocks).
   *
   * @return The entity count.
   */
  uint64 GetElementCount(void) const;

  /**
   * Creates a new command that operates on a subset of the entities this
   * command works on. The new instance uses the same logic as this instance
   * (that is, it is of the same type).
   *
   * @param offset       Zero-based index of the first entity to include in the
   *                     new command.
   * @param elementCount Number of elements to include.
   *
   * @return The new command.
   */
  ElementInitializerCommand& Slice(uint64 offset, uint64 elementCount) const;

  /**
   * Sets the memory grid layout to assume when selecting entity block address.
   *
   * @param blockLayout The block layout.
   * @param gridSize    Size of the grid.
   */
  void SetTargetGridLayout(
    const axis::services::memory::MemoryLayout& blockLayout, uint64 gridSize);
private:
  /**
   * Creates a new instance of this class by using the information provided.
   *
   * @param [in,out] associatedDataPtr Array of pointers pointing to the
   *                 data associated with each memory block (in the same order
   *                 as the memory blocks).
   * @param elementCount               Number of entities (blocks).
   *
   * @return The new command.
   */
  virtual ElementInitializerCommand& DoSlice(void **associatedDataPtr, 
    uint64 elementCount) const = 0;

  /**
   * Executes operations in a single entity data block.
   *
   * @param [in,out] blockDataAddress Base address of the data block.
   * @param externalMirroredAddress   Corresponding base address of the mirrored
   *                                  data block.
   * @param [in,out] associatedData   Pointer to the associated data for this
   *                 entity.
   */
  virtual void DoExecute(void *blockDataAddress,
                         const void *externalMirroredAddress,
                         void *associatedData) = 0;

  void **associatedDataPtr_;
  axis::services::memory::MemoryLayout *blockLayout_;
  uint64 elementCount_;
  uint64 gridSize_;
};

} } } } // namespace axis::services::memory::commands
