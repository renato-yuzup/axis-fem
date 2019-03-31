#pragma once
#include "FileStore.hpp"
#include <map>
#include "StreamWriter.hpp"

namespace axis { namespace services { namespace io {

class FileStore::FileStorePimpl
{
public:
  typedef std::map<axis::String, axis::services::io::StreamWriter *> file_set;
  file_set openedFiles;
  file_set tempFiles;
  axis::String basePath;
};

} } } // namespace axis::services::io
