#include "Formulation.hpp"
#include "domain/elements/FiniteElement.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"

namespace ade = axis::domain::elements;
namespace adf = axis::domain::formulations;
namespace afm = axis::foundation::memory;

adf::Formulation::Formulation( void )
{
	element_ = NULL;
}

adf::Formulation::~Formulation( void )
{
  element_ = NULL;
}

void adf::Formulation::UpdateGeometry( void )
{
  // nothing to do in base implementation
}

bool adf::Formulation::IsNonLinearFormulation( void ) const
{ // by default, formulations only handles linear case
  return false;
}

const ade::FiniteElement& adf::Formulation::Element( void ) const
{
	return *element_;
}

ade::FiniteElement& adf::Formulation::Element( void )
{
  return *element_;
}

void adf::Formulation::SetElement( ade::FiniteElement& element )
{
	element_ = &element;
}

bool adf::Formulation::IsCPUCapable( void ) const
{
  // Specialized implementation should explicitly indicate
  // that is NOT able to run in CPU
  return true;
}

bool adf::Formulation::IsGPUCapable( void ) const
{
  // Specialized implementation should explicitly indicate
  // that is able to run in GPU
  return false;
}

size_type adf::Formulation::GetGPUDataLength( void ) const
{
  return 10;
}

void adf::Formulation::InitializeGPUData( void *, real *artificialEnergy )
{
  *artificialEnergy = 0;
}

adf::FormulationStrategy& adf::Formulation::GetGPUStrategy( void )
{
  return FormulationStrategy::NullStrategy;
}

void * adf::Formulation::operator new( size_t bytes )
{
  // It is supposed that the finite element object will remain in memory
  // until the end of the program. That's why we discard the relative
  // pointer. We ignore the fact that an exception might occur in
  // constructor because if it does happen, the program will end.
  afm::RelativePointer ptr = System::GlobalMemory().Allocate(bytes);
  return *ptr;
}

void adf::Formulation::operator delete( void *ptr )
{
  // Since the relative pointer was discarded, we can't discard memory.
  // If it is really necessary, to free up resources, obliterating
  // memory pool is a better approach.
}
