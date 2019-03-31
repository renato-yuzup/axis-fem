#pragma once
#include "WorkbookFactoryLocator.hpp"
#include <set>

namespace axis{ namespace application { namespace locators {

class WorkbookFactoryLocator::Pimpl
{
public:
  typedef std::set<axis::application::factories::workbooks::WorkbookFactory *> factory_set;
  factory_set factories;
};

} } } // namespace axis::application::locators
