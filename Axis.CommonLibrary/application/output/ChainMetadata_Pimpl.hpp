#pragma once
#include "ChainMetadata.hpp"
#include <vector>
#include "AxisString.hpp"

namespace axis { namespace application { namespace output {

class ChainMetadata::Pimpl
{
public:
  typedef std::vector<axis::String> collector_list;

  axis::String Title;
  axis::String FilePath;
  axis::String Description;
  collector_list Collectors;
  bool IsAppend;
};

} } } // namespace axis::application::output
