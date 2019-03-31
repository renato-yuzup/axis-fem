#include "WorkFolder.hpp"
#include "WorkFolder_Pimpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/NotImplementedException.hpp"

namespace aaj = axis::application::jobs;
namespace asi = axis::services::io;
using axis::String;

aaj::WorkFolder::WorkFolder( const String& workingFolderPath )
{
  pimpl_ = new Pimpl();
  pimpl_->store = new asi::FileStore(workingFolderPath);
  pimpl_->workFolderLocation = workingFolderPath;
}

aaj::WorkFolder::~WorkFolder( void )
{
  pimpl_->store->ReleaseAll();
  pimpl_->store->Destroy();
  delete pimpl_;
}

void aaj::WorkFolder::Destroy( void ) const
{
  delete this;
}

size_type aaj::WorkFolder::StreamCount( void ) const
{
  return pimpl_->store->BorrowedCount();
}

bool aaj::WorkFolder::UsesStream( const String& streamPath ) const
{
  return !pimpl_->store->IsAvailable(streamPath);
}

bool aaj::WorkFolder::UsesStream( const String& streamPath, const String& basePath ) const
{
  return !pimpl_->store->IsAvailable(streamPath, basePath);
}

asi::StreamWriter& aaj::WorkFolder::GetOrCreateWorkFile( const String& streamPath )
{
  return pimpl_->store->AcquireStream(streamPath);
}

asi::StreamWriter& aaj::WorkFolder::GetOrCreateWorkFile( const String& streamPath, const String& basePath )
{
  return pimpl_->store->AcquireStream(streamPath, basePath);
}

asi::StreamWriter& aaj::WorkFolder::GetWorkFile( const String& streamPath )
{
  return pimpl_->store->GetStream(streamPath);
}

asi::StreamWriter& aaj::WorkFolder::GetWorkFile( const String& streamPath, const String& basePath )
{
  return pimpl_->store->GetStream(streamPath, basePath);
}

axis::String aaj::WorkFolder::GetLocation( void ) const
{
  return pimpl_->workFolderLocation;
}

asi::StreamWriter& aaj::WorkFolder::CreateTempFile( void )
{
  return CreateTempFile(_T("tmp"), _T("tempfile"));
}

asi::StreamWriter& aaj::WorkFolder::CreateTempFile( const axis::String& prefix )
{
  return CreateTempFile(prefix, _T("tempfile"));
}

asi::StreamWriter& aaj::WorkFolder::CreateTempFile( const String& prefix, const String& extension )
{
  axis::String fileName = pimpl_->GenerateTempFileName(GetLocation(), prefix, extension);
  asi::StreamWriter& tempFileStream = pimpl_->store->CreateTempStream(fileName);
  return tempFileStream;
}

asi::StreamReader& aaj::WorkFolder::OpenTempFileForRead( const axis::String& tempFileName )
{
  return pimpl_->store->OpenTempStream(tempFileName);
}

void aaj::WorkFolder::ClearTempFiles( void )
{
  pimpl_->store->ReleaseTempStreams();
}
