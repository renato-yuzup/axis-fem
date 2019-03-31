#include "SolverFactoryLocator.hpp"
#include "SolverFactoryLocator_Pimpl.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "services/management/ServiceLocator.hpp"

namespace aafal = axis::application::factories::algorithms;
namespace aal = axis::application::locators;
namespace adal = axis::domain::algorithms;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;

aal::SolverFactoryLocator::SolverFactoryLocator( void )
{
	pimpl_ = new Pimpl();
  pimpl_->parentCatalog = NULL;
}

aal::SolverFactoryLocator::~SolverFactoryLocator( void )
{
  // destroy all factories
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafal::SolverFactory& f = **it;
    f.Destroy();
  }
	delete pimpl_;
  pimpl_ = NULL;
}

bool aal::SolverFactoryLocator::CanBuild( const axis::String& solverTypeName, 
                                          const aslse::ParameterList& params, 
                                          real stepStartTime, 
                                          real stepEndTime ) const
{
  // search for a factory able to build a parser for this solver
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafal::SolverFactory& f = **it;
    if (f.CanBuild(solverTypeName, params, stepStartTime, stepEndTime))
    {
      return true;
    }
  }
  return false;
}

bool aal::SolverFactoryLocator::CanBuild( const axis::String& solverTypeName, 
                                          const aslse::ParameterList& params, 
                                          real stepStartTime, 
                                          real stepEndTime, 
                                          const axis::String& clockworkTypeName, 
                                          const aslse::ParameterList& clockworkParams ) const
{
  // search for a factory able to build a parser for this solver
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafal::SolverFactory& f = **it;
    if (f.CanBuild(solverTypeName, params, stepStartTime, stepEndTime, clockworkTypeName, clockworkParams))
    {
      return true;
    }
  }
  return false;
}

adal::Solver& aal::SolverFactoryLocator::BuildSolver( const axis::String& solverTypeName, 
                                                      const aslse::ParameterList& params, 
                                                      real stepStartTime, 
                                                      real stepEndTime )
{
  // search for a factory able to build a parser for this solver
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafal::SolverFactory& f = **it;
    if (f.CanBuild(solverTypeName, params, stepStartTime, stepEndTime))
    {	// found it, return parser
      return f.BuildSolver(solverTypeName, params, stepStartTime, stepEndTime);
    }
  }
  // unable to build this solver
  throw axis::foundation::InvalidOperationException(_T("Unable to build this type of solver."));
}

adal::Solver& aal::SolverFactoryLocator::BuildSolver( const axis::String& solverTypeName, 
                                                      const aslse::ParameterList& params, 
                                                      real stepStartTime, 
                                                      real stepEndTime, 
                                                      adal::Clockwork& clockwork )
{
  // search for a factory able to build a parser for this solver
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafal::SolverFactory& f = **it;
    if (f.CanBuild(solverTypeName, params, stepStartTime, stepEndTime))
    {	// found it, return parser
      return f.BuildSolver(solverTypeName, params, stepStartTime, stepEndTime, clockwork);
    }
  }

  // unable to build this solver
  throw axis::foundation::InvalidOperationException(_T("Unable to build this type of solver."));
}

void aal::SolverFactoryLocator::RegisterFactory( aafal::SolverFactory& factory )
{
  if (pimpl_->factories.find(&factory) != pimpl_->factories.end())
  {
    throw axis::foundation::ArgumentException(_T("Cannot register the same factory more than once."));
  }
  pimpl_->factories.insert(&factory);
}

void aal::SolverFactoryLocator::UnregisterFactory( aafal::SolverFactory& factory )
{
  if (pimpl_->factories.find(&factory) == pimpl_->factories.end())
  {
    throw axis::foundation::ArgumentException(_T("Factory not found."));
  }
  pimpl_->factories.erase(&factory);
}

const char * aal::SolverFactoryLocator::GetFeaturePath( void ) const
{
  return asmg::ServiceLocator::GetSolverLocatorPath();
}

const char * aal::SolverFactoryLocator::GetFeatureName( void ) const
{
  return "StandardSolverFactoryLocator";
}

void aal::SolverFactoryLocator::PostProcessRegistration( asmg::GlobalProviderCatalog& manager )
{
  pimpl_->parentCatalog = &manager;
}
