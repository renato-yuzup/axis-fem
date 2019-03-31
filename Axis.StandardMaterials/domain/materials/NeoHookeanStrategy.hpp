#pragma once
#include "domain/materials/MaterialStrategy.hpp"
#include "neohookean_commands/NeoHookeanStressCommand.hpp"

namespace axis { namespace domain { namespace materials {

class NeoHookeanStrategy : public MaterialStrategy
{
public:
  ~NeoHookeanStrategy(void);
  virtual UpdateStressCommand& GetUpdateStressCommand( void );
  static NeoHookeanStrategy& GetInstance(void);
private:
  NeoHookeanStrategy(void);

  neohookean_commands::NeoHookeanStressCommand stressCmd_;
  static NeoHookeanStrategy *instance_;
};

} } } // namespace axis::domain::materials
