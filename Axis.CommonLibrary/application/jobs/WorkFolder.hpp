#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "AxisString.hpp"
#include "services/io/StreamReader.hpp"
#include "services/io/StreamWriter.hpp"

namespace axis { namespace application { 	namespace jobs {

/**
 * Provides access to output files used by collectors.
**/
class AXISCOMMONLIBRARY_API WorkFolder
{
public:
  WorkFolder(const axis::String& workingFolderPath);
  ~WorkFolder(void);
  void Destroy(void) const;

  size_type StreamCount(void) const;
  bool UsesStream(const axis::String& streamPath) const;
  bool UsesStream(const axis::String& streamPath, const axis::String& basePath) const;

  axis::services::io::StreamWriter& GetOrCreateWorkFile(const axis::String& streamPath);
  axis::services::io::StreamWriter& GetOrCreateWorkFile(const axis::String& streamPath, const axis::String& basePath);
  axis::services::io::StreamWriter& GetWorkFile(const axis::String& streamPath);
  axis::services::io::StreamWriter& GetWorkFile(const axis::String& streamPath, const axis::String& basePath);

  axis::services::io::StreamWriter& CreateTempFile(void);
  axis::services::io::StreamWriter& CreateTempFile(const axis::String& prefix);
  axis::services::io::StreamWriter& CreateTempFile(const axis::String& prefix, const axis::String& extension);
  axis::services::io::StreamReader& OpenTempFileForRead(const axis::String& tempFileName);
  void ClearTempFiles(void);

  axis::String GetLocation(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};
} } } // namespace axis::application::jobs

