#pragma once
#include "MaterialFactoryLocator.hpp"
#include <set>
#include "application/factories/materials/MaterialFactory.hpp"

namespace axis { namespace application { namespace locators {

class MaterialFactoryLocator::Pimpl
{
public:
  typedef std::set<axis::application::factories::materials::MaterialFactory *> factory_set;
  factory_set factories;
};

} } } // namespace axis::application::locators
