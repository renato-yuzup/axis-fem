#pragma once
#include "ConstraintParserLocator.hpp"
#include <set>
#include "application/factories/boundary_conditions/ConstraintFactory.hpp"

namespace axis { namespace application { namespace locators {

class ConstraintParserLocator::Pimpl
{
public:
  typedef std::set<axis::application::factories::boundary_conditions::ConstraintFactory *> builder_set;
  typedef builder_set::iterator builder_iterator;
  builder_set builders;
};

} } } // namespace axis::application::locators
