#pragma once

namespace axis {   

namespace application { 

namespace jobs {
class StructuralAnalysis;
} // namespace jobs

namespace parsing { namespace core {
class ParseContext;
} } // namespace parsing::core

} // namespace application

namespace services { namespace language { 
namespace parsing {
  class ParseResult;
} // namespace parsing

namespace iterators {
  class InputIterator;
} // namespace iterators
} } // namespace services::language

namespace domain { namespace analyses {
class NumericalModel;
} } // namespace domain::analyses

} // namespace axis