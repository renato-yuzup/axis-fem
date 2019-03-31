#include "FlagCollection.hpp"
#include "FlagCollection_Pimpl.hpp"

using axis::domain::collections::FlagCollection;

FlagCollection::FlagCollection( void )
{
  pimpl_ = new Pimpl();
}

FlagCollection::~FlagCollection( void )
{
	delete pimpl_;
}


void FlagCollection::Add( const axis::String& flag )
{
  pimpl_->flags.insert(flag);
}

void FlagCollection::Remove( const axis::String& flag )
{
  pimpl_->flags.erase(flag);
}

bool FlagCollection::IsDefined( const axis::String& flag ) const
{
  return pimpl_->flags.find(flag) != pimpl_->flags.end();
}

void FlagCollection::Clear( void )
{
  pimpl_->flags.clear();
}

unsigned int FlagCollection::Count( void ) const
{
  return pimpl_->flags.size();
}

FlagCollection& FlagCollection::Clone( void ) const
{
  FlagCollection& col = *new FlagCollection();
  col.pimpl_->flags.insert(pimpl_->flags.begin(), pimpl_->flags.end());
  return col;
}

void FlagCollection::Destroy( void ) const
{
  delete this;
}
