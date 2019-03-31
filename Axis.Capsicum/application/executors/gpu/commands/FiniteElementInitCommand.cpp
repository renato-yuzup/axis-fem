#include "FiniteElementInitCommand.hpp"
#include "domain/elements/ElementGeometry.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "foundation/memory/pointer.hpp"
#include "services/memory/gpu/EBlockLayout.hpp"

namespace aaegc = axis::application::executors::gpu::commands;
namespace asmg  = axis::services::memory::gpu;
namespace ade   = axis::domain::elements;
namespace afm   = axis::foundation::memory;
namespace asmc  = axis::services::memory::commands;

aaegc::FiniteElementInitCommand::FiniteElementInitCommand( 
  void **associatedDataPtr, uint64 elementCount ) :
ElementInitializerCommand(associatedDataPtr, elementCount)
{
  // nothing to do here
}

aaegc::FiniteElementInitCommand::~FiniteElementInitCommand( void )
{
  // nothing to do here
}

asmc::ElementInitializerCommand& aaegc::FiniteElementInitCommand::DoSlice( 
  void **associatedDataPtr, uint64 elementCount ) const
{
  return *new FiniteElementInitCommand(associatedDataPtr, elementCount);
}

void aaegc::FiniteElementInitCommand::DoExecute( void *blockDataAddress,
  const void *, void *associatedData )
{
  ade::FiniteElement& element = *(ade::FiniteElement *)associatedData;
  uint64 elementId = (uint64)element.GetInternalId();
  int totalDofCount = element.Geometry().GetTotalDofCount();
  size_type formulationBlockSize = element.GetFormulationBlockSize();
  size_type materialBlockSize = element.GetMaterialBlockSize();
  asmg::EBlockLayout layout(formulationBlockSize, materialBlockSize);
  layout.InitMemoryBlock(blockDataAddress, elementId, totalDofCount);
  element.InitializeGPUFormulation(
    layout.GetFormulationBlockAddress(blockDataAddress));
  element.InitializeGPUMaterial(
    layout.GetMaterialBlockAddress(blockDataAddress));
}
