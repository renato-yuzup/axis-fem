#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "AxisString.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/management/Provider.hpp"
#include "application/factories/algorithms/SolverFactory.hpp"
#include "domain/algorithms/Solver.hpp"

namespace axis { namespace application { namespace locators {

class AXISPHYSALIS_API SolverFactoryLocator : public axis::services::management::Provider
{
public:
  SolverFactoryLocator(void);
  ~SolverFactoryLocator(void);
  bool CanBuild(const axis::String& solverTypeName, 
                const axis::services::language::syntax::evaluation::ParameterList& params, 
                real stepStartTime, real stepEndTime) const;
  bool CanBuild(const axis::String& solverTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& params, 
    real stepStartTime, real stepEndTime, const axis::String& clockworkTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& clockworkParams) const;
  axis::domain::algorithms::Solver& BuildSolver(const axis::String& solverTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& params, 
    real stepStartTime, real stepEndTime);
  axis::domain::algorithms::Solver& BuildSolver(const axis::String& solverTypeName, 
    const axis::services::language::syntax::evaluation::ParameterList& params, 
    real stepStartTime, real stepEndTime, axis::domain::algorithms::Clockwork& clockwork);
  void RegisterFactory(axis::application::factories::algorithms::SolverFactory& factory);
  void UnregisterFactory(axis::application::factories::algorithms::SolverFactory& factory);
  virtual const char * GetFeaturePath( void ) const;
  virtual const char * GetFeatureName( void ) const;
  virtual void PostProcessRegistration( axis::services::management::GlobalProviderCatalog& manager );
private:
  class Pimpl;
  Pimpl *pimpl_;
};	

} } } // namespace axis::application::locators
