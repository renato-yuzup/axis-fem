#include "ChainMetadata.hpp"
#include "ChainMetadata_Pimpl.hpp"

namespace aao = axis::application::output;

aao::ChainMetadata::ChainMetadata( void )
{
  pimpl_ = new Pimpl();
  pimpl_->IsAppend = false;
}

aao::ChainMetadata::ChainMetadata( const axis::String& title, const axis::String& fileName )
{
  pimpl_ = new Pimpl();
  pimpl_->Title = title;
  pimpl_->FilePath = fileName;
  pimpl_->IsAppend = false;
}

aao::ChainMetadata::ChainMetadata( const axis::String& title, const axis::String& fileName, 
                                   const axis::String& description )
{
  pimpl_ = new Pimpl();
  pimpl_->Title = title;
  pimpl_->FilePath = fileName;
  pimpl_->Description = description;
  pimpl_->IsAppend = false;
}

aao::ChainMetadata::ChainMetadata( const ChainMetadata& other )
{
  pimpl_ = new Pimpl();
  Copy(other);
}

aao::ChainMetadata::~ChainMetadata( void )
{
  delete pimpl_;
  pimpl_ = NULL;
}

aao::ChainMetadata& aao::ChainMetadata::operator =( const ChainMetadata& other )
{
  Copy(other);
  return *this;
}

axis::String aao::ChainMetadata::GetTitle( void ) const
{
  return pimpl_->Title;
}

axis::String aao::ChainMetadata::GetOutputFileName( void ) const
{
  return pimpl_->FilePath;
}

axis::String aao::ChainMetadata::GetShortDescription( void ) const
{
  return pimpl_->Description;
}

int aao::ChainMetadata::GetCollectorCount( void ) const
{
  return (int)pimpl_->Collectors.size();
}

axis::String aao::ChainMetadata::operator[]( int index ) const
{
  return pimpl_->Collectors[index];
}

axis::String aao::ChainMetadata::GetCollectorDescription( int index ) const
{
  return operator [](index);
}

bool aao::ChainMetadata::WillAppendData( void ) const
{
  return pimpl_->IsAppend;
}

void aao::ChainMetadata::AddCollectorDescription( const axis::String& description )
{
  pimpl_->Collectors.push_back(description);
}

void aao::ChainMetadata::SetTitle( const axis::String& title ) const
{
  pimpl_->Title = title;
}

void aao::ChainMetadata::SetOutputFileName( const axis::String& filename ) const
{
  pimpl_->FilePath = filename;
}

void aao::ChainMetadata::SetShortDescription( const axis::String& description ) const
{
  pimpl_->Description = description;
}

void aao::ChainMetadata::Copy( const ChainMetadata& other )
{
  pimpl_->Title = other.pimpl_->Title;
  pimpl_->FilePath = other.pimpl_->FilePath;
  pimpl_->Description = other.pimpl_->Description;
  pimpl_->IsAppend = other.pimpl_->IsAppend;
  pimpl_->Collectors.clear();
  Pimpl::collector_list::const_iterator end = other.pimpl_->Collectors.end();
  for (Pimpl::collector_list::const_iterator it = other.pimpl_->Collectors.begin(); it != end; ++it)
  {
    pimpl_->Collectors.push_back(*it);
  }
}

void aao::ChainMetadata::SetAppendDataState( bool state )
{
  pimpl_->IsAppend = state;
}
