#pragma once
#include "Foundation/Axis.CommonLibrary.hpp"
#include "UpdateStressCommand.hpp"

namespace axis { namespace domain { namespace materials {

class AXISCOMMONLIBRARY_API MaterialStrategy
{
public:
  MaterialStrategy(void);
  virtual ~MaterialStrategy(void);
  virtual UpdateStressCommand& GetUpdateStressCommand(void) = 0;
  static MaterialStrategy& NullStrategy;
};

} } } // namespace axis::domain::materials
