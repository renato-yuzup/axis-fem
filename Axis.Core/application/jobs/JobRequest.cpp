#include "JobRequest.hpp"
#include "JobRequest_Pimpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/OutOfBoundsException.hpp"

using axis::application::jobs::JobRequest;

JobRequest::JobRequest( const axis::String& masterFilename, const axis::String& baseIncludePath, const axis::String& outputFolderPath )
{
  pimpl_ = new Pimpl();
  pimpl_->masterFilename = masterFilename;
  pimpl_->outputFolderPath = outputFolderPath;
  pimpl_->baseIncludePath = baseIncludePath;
}

JobRequest::~JobRequest( void )
{
  pimpl_->flags.clear();
	delete pimpl_;
}

void JobRequest::Destroy( void ) const
{
  delete this;
}

axis::String JobRequest::GetMasterInputFilePath( void ) const
{
  return pimpl_->masterFilename;
}

axis::String JobRequest::GetBaseIncludePath( void ) const
{
  return pimpl_->baseIncludePath;
}

axis::String JobRequest::GetOutputFolderPath( void ) const
{
  return pimpl_->outputFolderPath;
}

void JobRequest::AddConditionalFlag( const axis::String& flagName )
{
  // search for a possible duplicate
  Pimpl::flag_set::nth_index<0>::type& uniqueIndex = pimpl_->flags.get<0>();
  if (uniqueIndex.find(flagName) != uniqueIndex.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->flags.insert(flagName);
}

void JobRequest::ClearConditionalFlags( void )
{
  pimpl_->flags.clear();
}

size_type JobRequest::GetConditionalFlagsCount( void ) const
{
  return (size_type)pimpl_->flags.size();
}

axis::String JobRequest::GetConditionalFlag( size_type index ) const
{
  if (index >= pimpl_->flags.size())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  const Pimpl::flag_set::nth_index<1>::type& randomIndex = 
      pimpl_->flags.get<1>();
  return randomIndex[index];
}

JobRequest& JobRequest::Clone( void ) const
{
  JobRequest& clone = *new JobRequest(pimpl_->masterFilename,
                                      pimpl_->baseIncludePath,
                                      pimpl_->outputFolderPath);
  // search for a possible duplicate
  Pimpl::flag_set::nth_index<0>::type& uniqueIndex = pimpl_->flags.get<0>();
  Pimpl::flag_set::nth_index<0>::type::iterator end = uniqueIndex.end();
  for (Pimpl::flag_set::nth_index<0>::type::iterator it = uniqueIndex.begin();
       it != end; ++it)
  {
    clone.AddConditionalFlag(*it);
  }
  return clone;
}

