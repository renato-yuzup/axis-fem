#pragma once
#include "WorkFolder.hpp"
#include <map>
#include <boost/filesystem.hpp>
#include "services/io/FileStore.hpp"
#include "services/io/StreamWriter.hpp"
#include "foundation/uuids/Uuid.hpp"

namespace axis { namespace application { namespace jobs {

class WorkFolder::Pimpl
{
public:
  axis::String workFolderLocation;
  axis::services::io::FileStore *store;

  axis::String GenerateTempFileName(const axis::String& basePath, const axis::String& prefix, const axis::String& extension) const
  {
    axis::String generatedFileName;
    boost::filesystem::path generatedFilePathObj;
    boost::filesystem::path basePathObj(basePath);

    do
    {
      axis::foundation::uuids::Uuid gen = axis::foundation::uuids::Uuid::GenerateRandom();
      generatedFileName = prefix + gen.ToStringAsByteSequence() + _T(".") + extension;
      generatedFilePathObj = basePathObj;
      generatedFilePathObj += boost::filesystem::path(generatedFileName);
    } while (boost::filesystem::exists(generatedFilePathObj));

    return generatedFileName;
  }
};

} } } // axis::application::jobs

