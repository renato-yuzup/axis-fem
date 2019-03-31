#pragma once
#include "services/memory/commands/ElementInitializerCommand.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace commands {

/**
* Executes initialization procedures for finite element objects. Because
* it is expected that initialization occurs in the host, execution must be
* done prior to mirroring memory to external processing device.
 */
class FiniteElementInitCommand : 
  public axis::services::memory::commands::ElementInitializerCommand
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] associatedDataPtr Array of pointers pointing to 
   *                 corresponding finite element objects to be initialized.
   * @param curveCount                 Number of finite element objects to 
   *                                   initialize.
   */
  FiniteElementInitCommand(void **associatedDataPtr, uint64 elementCount);
  virtual ~FiniteElementInitCommand(void);
private:
  virtual axis::services::memory::commands::ElementInitializerCommand& DoSlice( 
    void **associatedDataPtr, uint64 elementCount ) const;
  virtual void DoExecute( void *blockDataAddress,
    const void *externalMirroredAddress, void *associatedData );
};

} } } } } // namespace axis::application::executors::gpu::commands
