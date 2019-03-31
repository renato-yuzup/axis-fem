#pragma once
#include "CurveUpdateCommand.hpp"

namespace axis { namespace domain { namespace curves {

class NullCurveUpdateCommand : public CurveUpdateCommand
{
public:
  NullCurveUpdateCommand(void);
  ~NullCurveUpdateCommand(void);
  virtual void Run( uint64, uint64, void *, const axis::Dimension3D&, 
    const axis::Dimension3D&, void * );
  static CurveUpdateCommand& GetInstance(void);
};

} } } // namespace axis::domain::curves
