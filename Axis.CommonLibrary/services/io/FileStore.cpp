#include "FileStore.hpp"
#include "FileStorePimpl.hpp"
#include <boost/filesystem.hpp>
#include "services/io/FileSystem.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/IOException.hpp"
#include "BinaryWriter.hpp"
#include "BinaryReader.hpp"

namespace asio = axis::services::io;

namespace {
  axis::String ToCanonicalPath( const axis::String& pathName, const axis::String& basePath )
  {
    // check if pathname needs to be converted into absolute path
    axis::String absolutePath = pathName;
    if (!asio::FileSystem::IsAbsolutePath(absolutePath))
    {
      absolutePath = asio::FileSystem::ConcatenatePath(basePath, absolutePath);
    }
    return asio::FileSystem::ToCanonicalPath(absolutePath);
  }
} // namespace

asio::FileStore::FileStore( const axis::String& basePath )
{
  pimpl_ = new FileStorePimpl();
  // check if it is a valid path
  if (!asio::FileSystem::IsDirectory(basePath))
  {
    throw axis::foundation::ArgumentException(_T("basePath"));
  }
  pimpl_->basePath = basePath;
}

asio::FileStore::~FileStore( void )
{
  ReleaseAll();
  delete pimpl_;
}

size_type asio::FileStore::BorrowedCount( void ) const
{
  return (size_type)pimpl_->openedFiles.size();
}

asio::StreamWriter& asio::FileStore::AcquireStream( const axis::String& fileName )
{
  return AcquireStream(fileName, pimpl_->basePath);
}

asio::StreamWriter& asio::FileStore::AcquireStream( const axis::String& fileName, const axis::String& basePath)
{
  if (!IsAvailable(fileName))
  {
    throw axis::foundation::IOException(_T("File system resource in use."));
  }
  axis::String canonicalPath = ToCanonicalPath(fileName, basePath);
  canonicalPath.trim().to_lower_case();

  asio::StreamWriter& writer = BinaryWriter::Create(canonicalPath);

  pimpl_->openedFiles.insert(FileStorePimpl::file_set::value_type(canonicalPath, &writer));

  return writer;
}

asio::StreamWriter& asio::FileStore::GetStream( const axis::String& fileName )
{
  return GetStream(fileName, pimpl_->basePath);
}

asio::StreamWriter& asio::FileStore::GetStream( const axis::String& fileName, const axis::String& basePath )
{
  if (IsAvailable(fileName))
  {
    throw axis::foundation::IOException(_T("Stream has not been acquired yet."));
  }
  axis::String canonicalPath = ToCanonicalPath(fileName, basePath);
  canonicalPath.trim().to_lower_case();

  return *pimpl_->openedFiles[canonicalPath];
}

void asio::FileStore::ReleaseStream( asio::StreamWriter& writer )
{
  ReleaseStream(writer.GetStreamPath());
}

void asio::FileStore::ReleaseStream( const axis::String& fileName )
{
  ReleaseStream(fileName, pimpl_->basePath);
}

void asio::FileStore::ReleaseStream( const axis::String& fileName, const axis::String& basePath )
{
  axis::String canonicalPath = ToCanonicalPath(fileName, basePath);
  canonicalPath.trim().to_lower_case();

  if (IsAvailable(canonicalPath))
  {
    throw axis::foundation::IOException(_T("File system resource not found."));
  }

  // close and destroy stream
  asio::StreamWriter *writer = pimpl_->openedFiles[canonicalPath];
  if (writer->IsOpen()) writer->Close();
  writer->Destroy();

  pimpl_->openedFiles.erase(canonicalPath);
}

void asio::FileStore::ReleaseAll( void )
{
  FileStorePimpl::file_set::iterator end = pimpl_->openedFiles.end();
  for (FileStorePimpl::file_set::iterator it = pimpl_->openedFiles.begin(); it != end; ++it)
  {
    asio::StreamWriter *writer = it->second;
    if (writer->IsOpen()) writer->Close();
    writer->Destroy();
  }
  pimpl_->openedFiles.clear();
}

bool asio::FileStore::IsAvailable( const axis::String& fileName ) const
{
  return IsAvailable(fileName, pimpl_->basePath);
}

bool asio::FileStore::IsAvailable( const axis::String& fileName, const axis::String& basePath ) const
{
  // first, check if it resolves to a resource with same key
  axis::String canonicalPath = ToCanonicalPath(fileName, pimpl_->basePath);
  canonicalPath.trim().to_lower_case();
  if (pimpl_->openedFiles.find(canonicalPath) != pimpl_->openedFiles.end())
  {
    return false;
  }

  // now, check for every opened file if this canonical path resolves to the
  // same filename with which was originally requested
  FileStorePimpl::file_set::const_iterator end = pimpl_->openedFiles.end();
  for (FileStorePimpl::file_set::const_iterator it = pimpl_->openedFiles.begin(); it != end; ++it)
  {
    axis::String originalPath = it->first;
    if (asio::FileSystem::IsSameFile(canonicalPath, originalPath))
    {
      return false;
    }
  }
  // no equivalences were found, it is available
  return true;
}

axis::String asio::FileStore::GetBaseSearchPath( void ) const
{
  return pimpl_->basePath;
}

void asio::FileStore::Destroy( void ) const
{
  delete this;
}

void asio::FileStore::PrepareAll( void )
{
  FileStorePimpl::file_set::iterator end = pimpl_->openedFiles.end();
  for (FileStorePimpl::file_set::iterator it = pimpl_->openedFiles.begin(); it != end; ++it)
  {
    asio::StreamWriter *writer = it->second;
    if (!writer->IsOpen()) writer->Open();
  }
}

asio::StreamWriter& asio::FileStore::CreateTempStream( const axis::String& fileName )
{
  axis::String canonicalPath = ToCanonicalPath(fileName, pimpl_->basePath);
  canonicalPath.trim().to_lower_case();
  if (pimpl_->tempFiles.find(canonicalPath) != pimpl_->tempFiles.end())
  {
    throw axis::foundation::IOException(_T("File system resource in use."));
  }

  asio::StreamWriter& writer = BinaryWriter::Create(canonicalPath);
  pimpl_->tempFiles[canonicalPath] = &writer;
  return writer;
}

asio::StreamReader& asio::FileStore::OpenTempStream( const axis::String& fileName )
{
  axis::String canonicalPath = ToCanonicalPath(fileName, pimpl_->basePath);
  canonicalPath.trim().to_lower_case();
  if (pimpl_->tempFiles.find(canonicalPath) == pimpl_->tempFiles.end())
  {
    throw axis::foundation::IOException(_T("Temp file is out of scope or does not exist."));
  }

  asio::StreamReader& reader = BinaryReader::Create(canonicalPath);
  return reader;
}

void asio::FileStore::ReleaseTempStreams( void )
{
  while (!pimpl_->tempFiles.empty())
  {
    FileStorePimpl::file_set::iterator it = pimpl_->tempFiles.begin();
    asio::StreamWriter& tempFile = *it->second;
    if (tempFile.IsOpen()) tempFile.Close();  
    try
    {
      asio::FileSystem::PurgeFile(tempFile.GetStreamPath());
    }
    catch (...)
    { // ignore on error
    }
    tempFile.Destroy();
    pimpl_->tempFiles.erase(it);
  }
}
