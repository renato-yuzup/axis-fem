#pragma once
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output {

  /**
   * Values that specify data types.
   */
  enum AXISCOMMONLIBRARY_API DataType
  {
    kIntegerNumber,
    kRealNumber,
    kLiteral,
    kCharacter,
    kBooleanField,
    kUndefined,
    kMatrix,
    kVector,
  };

} } } // namespace axis::application::output
