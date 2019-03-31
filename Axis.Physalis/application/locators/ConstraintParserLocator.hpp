#pragma once
#include "foundation/Axis.Physalis.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include "application/factories/boundary_conditions/ConstraintFactory.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/jobs/StructuralAnalysis.hpp"

namespace axis { namespace application { namespace locators {

class AXISPHYSALIS_API ConstraintParserLocator : 
  public axis::application::factories::parsers::BlockProvider
{
public:
  ConstraintParserLocator(void);
	~ConstraintParserLocator(void);

	void RegisterFactory(axis::application::factories::boundary_conditions::ConstraintFactory& builder);
	void UnregisterFactory(axis::application::factories::boundary_conditions::ConstraintFactory& builder);

	axis::services::language::parsing::ParseResult TryParse(
        const axis::services::language::iterators::InputIterator& begin, 
        const axis::services::language::iterators::InputIterator& end);
	axis::services::language::parsing::ParseResult ParseAndBuild(
        axis::application::jobs::StructuralAnalysis& analysis, 
        axis::application::parsing::core::ParseContext& context, 
        const axis::services::language::iterators::InputIterator& begin, 
        const axis::services::language::iterators::InputIterator& end);
  virtual bool CanParse( const axis::String& blockName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
  virtual axis::application::parsing::parsers::BlockParser& BuildParser( 
    const axis::String& contextName, 
    const axis::services::language::syntax::evaluation::ParameterList& paramList );
  virtual const char * GetFeaturePath( void ) const;
  virtual const char * GetFeatureName( void ) const;
private:
  class Pimpl;
  Pimpl *pimpl_;
};

} } } // namespace axis::application::locators
