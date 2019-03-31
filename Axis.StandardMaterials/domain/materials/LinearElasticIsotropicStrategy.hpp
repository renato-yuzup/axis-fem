#pragma once
#include "domain/materials/MaterialStrategy.hpp"
#include "linear_iso_commands/LinearIsoStressCommand.hpp"

namespace axis { namespace domain { namespace materials {

class LinearElasticIsotropicStrategy : public MaterialStrategy
{
public:
  ~LinearElasticIsotropicStrategy(void);
  virtual UpdateStressCommand& GetUpdateStressCommand( void );

  static LinearElasticIsotropicStrategy& GetInstance(void);
private:
  LinearElasticIsotropicStrategy(void);

  linear_iso_commands::LinearIsoStressCommand stressCmd_;
  static LinearElasticIsotropicStrategy *instance_;
};

} } } // namespace axis::domain::materials
