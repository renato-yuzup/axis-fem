#pragma once
#include "domain/materials/MaterialStrategy.hpp"
#include "bilinear_plasticity_commands/BiLinearPlasticityStressCommand.hpp"

namespace axis { namespace domain { namespace materials {

class BiLinearPlasticityStrategy : public MaterialStrategy
{
public:
  BiLinearPlasticityStrategy(void);
  ~BiLinearPlasticityStrategy(void);
  virtual UpdateStressCommand& GetUpdateStressCommand( void );

  static BiLinearPlasticityStrategy& GetInstance(void);
private:
  bilinear_plasticity_commands::BiLinearPlasticityStressCommand stressCmd_;
  static BiLinearPlasticityStrategy *instance_;
};

} } } // namespace axis::domain::materials
