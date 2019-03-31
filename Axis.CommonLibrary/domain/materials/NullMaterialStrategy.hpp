#pragma once
#include "MaterialStrategy.hpp"

namespace axis { namespace domain { namespace materials {

class NullMaterialStrategy : public MaterialStrategy
{
public:
  NullMaterialStrategy(void);
  ~NullMaterialStrategy(void);
  virtual UpdateStressCommand& GetUpdateStressCommand(void);
private:
  UpdateStressCommand *stressCommand_;
};

} } } // namespace axis::domain::materials
