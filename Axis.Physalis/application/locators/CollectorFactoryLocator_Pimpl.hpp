#pragma once
#include "CollectorFactoryLocator.hpp"
#include <set>
#include "WorkbookFactoryLocator.hpp"

namespace axis { namespace application { namespace locators {

class CollectorFactoryLocator::Pimpl
{
public:
  typedef std::set<axis::application::factories::collectors::CollectorFactory *> factory_set;
  factory_set builders;
  axis::application::locators::WorkbookFactoryLocator *formatLocator;
};

} } } // namespace axis::application::locators
