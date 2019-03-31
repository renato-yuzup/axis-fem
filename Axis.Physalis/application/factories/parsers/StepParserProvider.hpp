#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/fwd/locators_fwd.hpp"
#include "application/factories/parsers/BlockProvider.hpp"

namespace axis { namespace application { namespace factories { namespace parsers {

class AXISPHYSALIS_API StepParserProvider : public axis::application::factories::parsers::BlockProvider
{
public:
	StepParserProvider(void);
	~StepParserProvider(void);

	virtual bool CanParse( const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
	virtual const char * GetFeaturePath( void ) const;
	virtual const char * GetFeatureName( void ) const;
protected:
  virtual void DoOnPostProcessRegistration( 
    axis::services::management::GlobalProviderCatalog& rootManager );
private:
  axis::application::locators::SolverFactoryLocator *solverLocator_;
  axis::application::locators::ClockworkFactoryLocator *clockworkLocator_;
  axis::application::locators::CollectorFactoryLocator *collectorLocator_;
  axis::application::locators::WorkbookFactoryLocator *formatLocator_;

  bool IsRequiredParametersPresentAndValid(
    const axis::services::language::syntax::evaluation::ParameterList& paramList) const;
};

} } } } // namespace axis::application::factories::parsers
