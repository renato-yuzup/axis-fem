#pragma once
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

/**
  * Defines capturing state of XX-tensor component.
  */
enum AXISCOMMONLIBRARY_API XXDirectionState
{
  kXXEnabled, kXXDisabled
};
/**
  * Defines capturing state of YY-tensor component.
  */
enum AXISCOMMONLIBRARY_API YYDirectionState
{
  kYYEnabled, kYYDisabled
};
/**
  * Defines capturing state of ZZ-tensor component.
  */
enum AXISCOMMONLIBRARY_API ZZDirectionState
{
  kZZEnabled, kZZDisabled
};
/**
  * Defines capturing state of XY-tensor component.
  */
enum AXISCOMMONLIBRARY_API XYDirectionState
{
  kXYEnabled, kXYDisabled
};
/**
  * Defines capturing state of YZ-tensor component.
  */
enum AXISCOMMONLIBRARY_API YZDirectionState
{
  kYZEnabled, kYZDisabled
};
/**
  * Defines capturing state of XZ-tensor component.
  */
enum AXISCOMMONLIBRARY_API XZDirectionState
{
  kXZEnabled, kXZDisabled
};

} } } } // namespace axis::application::output::collectors
