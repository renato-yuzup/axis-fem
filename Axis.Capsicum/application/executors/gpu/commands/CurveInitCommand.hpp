#pragma once
#include "services/memory/commands/ElementInitializerCommand.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace commands {

/**
* Executes initialization procedures for curve objects. Because
* it is expected that initialization occurs in the host, execution must be
* done prior to mirroring memory to external processing device.
 */
class CurveInitCommand : 
  public axis::services::memory::commands::ElementInitializerCommand
{
public:
  /**
   * Constructor.
   *
   * @param [in,out] associatedDataPtr Array of pointers pointing to 
   *                 corresponding curve objects to be initialized.
   * @param curveCount                 Number of curve objects to initialize.
   */
  CurveInitCommand(void **associatedDataPtr, uint64 curveCount);
  virtual ~CurveInitCommand(void);
private:
  virtual ElementInitializerCommand& DoSlice(void **associatedDataPtr, 
    uint64 elementCount ) const;
  virtual void DoExecute( void *blockDataAddress,
    const void *externalMirroredAddress,
    void *associatedData );
};

} } } } } // namespace axis::application::executors::gpu::commands
