#pragma once
#include "Sketchbook.hpp"
#include <map>
#include "SectionDefinition.hpp"
#include "AxisString.hpp"

namespace axis { namespace application { namespace parsing { namespace core {

class Sketchbook::Pimpl
{
public:
  typedef std::map<axis::String, 
      const axis::application::parsing::core::SectionDefinition *> section_set;
  section_set sections;
};

} } } } // namespace axis::application::parsing::core

