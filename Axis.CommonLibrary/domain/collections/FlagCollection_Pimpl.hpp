#pragma once
#include "FlagCollection.hpp"
#include <set>

namespace axis { namespace domain { namespace collections {
  
class FlagCollection::Pimpl
{
public:
  typedef std::set<axis::String> flag_set;
  flag_set flags;
};

} } } // namespace axis::domain::collections
