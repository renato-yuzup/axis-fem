#pragma once
#include "EntityLabeler.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class EntityLabeler::Pimpl
{
public:
  size_type nextNodeLabel;
  size_type nextElementLabel;
  size_type nextDofLabel;
};

} } } } // namespace axis::application::parsing::core
