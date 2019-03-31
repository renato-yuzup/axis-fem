#pragma once
#include "ElementParserLocator.hpp"
#include <set>
#include "../factories/parsers/ElementParserFactory.hpp"

namespace axis { namespace application { namespace locators {

class ElementParserLocator::Pimpl
{
public:
  typedef std::set<axis::application::factories::parsers::ElementParserFactory *> factory_set;
  factory_set factories;
};

} } } // namespace axis::application::locators

