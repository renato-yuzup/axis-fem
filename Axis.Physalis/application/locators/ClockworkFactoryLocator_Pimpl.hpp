#pragma once
#include <set>
#include "ClockworkFactoryLocator.hpp"

namespace axis { namespace application { namespace locators {

class ClockworkFactoryLocator::Pimpl
{
public:
  typedef std::set<
      axis::application::factories::algorithms::ClockworkFactory *> factory_set;
  typedef factory_set::iterator factory_iterator;
  factory_set factories;
};

} } } // namespace axis::application::locators
