#pragma once
#include "FileStreamWriter.hpp"
#include <fstream>
#include "AxisString.hpp"

namespace axis { namespace services { namespace io {

class FileStreamWriter::FileStreamWriterPimpl
{
public:
  typedef std::basic_ofstream<axis::String::char_type> ofstream;
  ofstream stream;
  axis::String fileName;
};

} } } // namespace axis::services::io
