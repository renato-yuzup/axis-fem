#include "MaterialModel.hpp"
#include "System.hpp"
#include "foundation/memory/HeapBlockArena.hpp"

namespace adm = axis::domain::materials;
namespace afm = axis::foundation::memory;

adm::MaterialModel::MaterialModel( real density, int pointCount )
{
	density_ = density;
  pointCount_ = pointCount;
}

adm::MaterialModel::~MaterialModel( void )
{
	// nothing to do here
}

real adm::MaterialModel::Density( void ) const
{
	return density_;
}

bool axis::domain::materials::MaterialModel::IsGPUCapable( void ) const
{
  // Specialized implementation should explicitly indicate
  // that is able to run in GPU
  return false;
}

bool axis::domain::materials::MaterialModel::IsCPUCapable( void ) const
{
  // Specialized implementation should explicitly indicate
  // that is NOT able to run in CPU
  return true;
}

size_type adm::MaterialModel::GetDataBlockSize( void ) const
{
  return 10;
}

void adm::MaterialModel::InitializeGPUData(void *, real *density, 
  real *, real *, real *, real *)
{
  *density = Density();
  // other members should be initialized by derived classes
}

#if !defined(AXIS_NO_MEMORY_ARENA)
void * adm::MaterialModel::operator new( size_t bytes )
{
  // It is supposed that the finite element object will remain in memory
  // until the end of the program. That's why we discard the relative
  // pointer. We ignore the fact that an exception might occur in
  // constructor because if it does happen, the program will end.
  afm::RelativePointer ptr = System::GlobalMemory().Allocate(bytes);
  return *ptr;
}

void adm::MaterialModel::operator delete( void *ptr )
{
  // Since the relative pointer was discarded, we can't discard memory.
  // If it is really necessary, to free up resources, obliterating
  // memory pool is a better approach.
}
#endif

adm::MaterialStrategy& adm::MaterialModel::GetGPUStrategy( void )
{
  return MaterialStrategy::NullStrategy;
}

int adm::MaterialModel::GetMaterialPointCount( void ) const
{
  return pointCount_;
}
