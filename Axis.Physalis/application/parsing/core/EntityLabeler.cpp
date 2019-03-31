#include "EntityLabeler.hpp"
#include "EntityLabeler_Pimpl.hpp"

namespace aapc = axis::application::parsing::core;

aapc::EntityLabeler::EntityLabeler( void )
{
  pimpl_ = new Pimpl();
  pimpl_->nextNodeLabel = 0;
  pimpl_->nextElementLabel = 0;
  pimpl_->nextDofLabel = 0;
}

aapc::EntityLabeler::~EntityLabeler( void )
{
  delete pimpl_;
}

void aapc::EntityLabeler::Destroy( void ) const
{
  delete this;
}

size_type aapc::EntityLabeler::PickNodeLabel( void )
{
  return pimpl_->nextNodeLabel++;
}

size_type aapc::EntityLabeler::PickElementLabel( void )
{
  return pimpl_->nextElementLabel++;
}

size_type aapc::EntityLabeler::PickDofLabel( void )
{
  return pimpl_->nextDofLabel++;
}

size_type aapc::EntityLabeler::GetGivenNodeLabelCount( void ) const
{
  return pimpl_->nextNodeLabel + 1;
}

size_type aapc::EntityLabeler::GetGivenElementLabelCount( void ) const
{
  return pimpl_->nextElementLabel + 1;
}

size_type aapc::EntityLabeler::GetGivenDofLabelCount( void ) const
{
  return pimpl_->nextDofLabel + 1;
}
