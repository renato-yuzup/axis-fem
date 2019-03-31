#pragma once
#include "foundation/computing/KernelCommand.hpp"
#include "Foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace domain { namespace curves {

class AXISCOMMONLIBRARY_API CurveUpdateCommand : public axis::foundation::computing::KernelCommand
{
public:
  CurveUpdateCommand(void);
  virtual ~CurveUpdateCommand(void);
  void SetTime(real time);
protected:
  real GetTime(void) const;
private:
  real time_;
};

} } } // namespace axis::domain::curves
