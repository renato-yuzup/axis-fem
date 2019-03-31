#pragma once
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

/**
  * Defines capturing state of X-spatial component.
  */
enum AXISCOMMONLIBRARY_API XDirectionState
{
  kXEnabled, kXDisabled
};

/**
  * Defines capturing state of Y-spatial component.
  */
enum AXISCOMMONLIBRARY_API YDirectionState
{
  kYEnabled, kYDisabled
};

/**
  * Defines capturing state of Z-spatial component.
  */
enum AXISCOMMONLIBRARY_API ZDirectionState
{
  kZEnabled, kZDisabled
};

} } } } // namespace axis::application::output::collectors
