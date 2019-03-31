#include "IntegrationPoint.hpp"
#include "domain/physics/InfinitesimalState.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/memory/pointer.hpp"

namespace adi = axis::domain::integration;
namespace adp = axis::domain::physics;
namespace afm = axis::foundation::memory;

adi::IntegrationPoint::IntegrationPoint( void )
{
	x_ = 0; y_ = 0; z_ = 0; weight_ = 1;
  state_ = adp::InfinitesimalState::Create();
}

adi::IntegrationPoint::IntegrationPoint( coordtype x, coordtype y ) 
{
  x_ = x; y_ = y; z_ = 0; weight_ = 1;
  state_ = adp::InfinitesimalState::Create();
}

adi::IntegrationPoint::IntegrationPoint( coordtype x, coordtype y, coordtype z )
{
  x_ = x; y_ = y; z_ = z; weight_ = 1;
  state_ = adp::InfinitesimalState::Create();
}

adi::IntegrationPoint::IntegrationPoint( coordtype x, coordtype y, coordtype z, real weight )
{
  x_ = x; y_ = y; z_ = z; weight_ = weight;
  state_ = adp::InfinitesimalState::Create();
}

adi::IntegrationPoint::~IntegrationPoint( void )
{
  absref<adp::InfinitesimalState>(state_).Destroy();
  System::ModelMemory().Deallocate(state_);
}

void adi::IntegrationPoint::Destroy( void ) const
{
	delete this;
}

adp::InfinitesimalState& adi::IntegrationPoint::State( void )
{
  return absref<adp::InfinitesimalState>(state_);
}

const adp::InfinitesimalState& adi::IntegrationPoint::State( void ) const
{
  return absref<adp::InfinitesimalState>(state_);
}

real& adi::IntegrationPoint::Weight( void )
{
  return weight_;
}

real adi::IntegrationPoint::Weight( void ) const
{
  return weight_;
}

real& adi::IntegrationPoint::X( void )
{
  return x_;
}

real adi::IntegrationPoint::X( void ) const
{
  return x_;
}

real& adi::IntegrationPoint::Y( void )
{
  return y_;
}

real adi::IntegrationPoint::Y( void ) const
{
  return y_;
}

real& adi::IntegrationPoint::Z( void )
{
  return z_;
}

real adi::IntegrationPoint::Z( void ) const
{
  return z_;
}

afm::RelativePointer adi::IntegrationPoint::Create( void )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(IntegrationPoint));
  new (*ptr) IntegrationPoint();
  return ptr;
}

afm::RelativePointer adi::IntegrationPoint::Create( coordtype x, coordtype y )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(IntegrationPoint));
  new (*ptr) IntegrationPoint(x, y);
  return ptr;
}

afm::RelativePointer adi::IntegrationPoint::Create( coordtype x, coordtype y, coordtype z )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(IntegrationPoint));
  new (*ptr) IntegrationPoint(x, y, z);
  return ptr;
}

afm::RelativePointer adi::IntegrationPoint::Create( coordtype x, coordtype y, coordtype z, real weight )
{
  afm::RelativePointer ptr = System::ModelMemory().Allocate(sizeof(IntegrationPoint));
  new (*ptr) IntegrationPoint(x, y, z, weight);
  return ptr;
}

void * adi::IntegrationPoint::operator new( size_t bytes )
{
  return ::operator new(bytes);
}

void * adi::IntegrationPoint::operator new( size_t, void *ptr )
{
  return ptr;
}

void adi::IntegrationPoint::operator delete( void *ptr )
{
  ::operator delete(ptr);
}

void adi::IntegrationPoint::operator delete( void *, void * )
{
  // Since we have discarded the relative pointer, there is no way we
  // can delete the point. Nothing to do here.
}
