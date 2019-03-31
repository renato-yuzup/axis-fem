#pragma once
#include "SolverFactoryLocator.hpp"
#include <set>
#include "services/management/GlobalProviderCatalog.hpp"
#include "application/factories/algorithms/SolverFactory.hpp"

namespace axis { namespace application { namespace locators {

class SolverFactoryLocator::Pimpl
{
public:
  typedef std::set<axis::application::factories::algorithms::SolverFactory *> factory_set;
  axis::services::management::GlobalProviderCatalog *parentCatalog;
  factory_set factories;
};

} } } // namespace axis::application::locators
