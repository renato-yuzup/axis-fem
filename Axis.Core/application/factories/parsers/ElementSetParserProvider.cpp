#include "ElementSetParserProvider.hpp"
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include "application/parsing/parsers/ElementSetParser.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/management/ServiceLocator.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;
namespace afd = axis::foundation::definitions;

aafp::ElementSetParserProvider::ElementSetParserProvider(void)
{
	// nothing to do
}

aafp::ElementSetParserProvider::~ElementSetParserProvider(void)
{
	// nothing to do
}

bool aafp::ElementSetParserProvider::CanParse( const axis::String& blockName, 
                                               const aslse::ParameterList& params )
{
	bool ok =	(blockName.compare(afd::AxisInputLanguage::ElementSetSyntax.BlockName) == 0 && 
				params.IsDeclared(afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName));
	if (ok) return 
    params.GetParameterValue(afd::AxisInputLanguage::ElementSetSyntax.SetIdAttributeName).IsAtomic();
	return false;
}

void aafp::ElementSetParserProvider::RegisterProvider( aafp::BlockProvider& provider )
{
	// fail
	throw axis::foundation::NotSupportedException();
}

void aafp::ElementSetParserProvider::UnregisterProvider( aafp::BlockProvider& provider )
{
	// fail
	throw axis::foundation::NotSupportedException();
}

bool aafp::ElementSetParserProvider::IsLeaf( void )
{
	// we won't allow any nested blocks here
	return true;	
}

const char * aafp::ElementSetParserProvider::GetFeaturePath( void ) const
{
	return asmg::ServiceLocator::GetElementSetInputParserProviderPath();
}

const char * aafp::ElementSetParserProvider::GetFeatureName( void ) const
{
	return "StandardElementSetParserProvider";
}

aapps::BlockParser& aafp::ElementSetParserProvider::BuildParser( const axis::String& contextName, 
                                                                 const aslse::ParameterList& params )
{
	// get params
	return *new aapps::ElementSetParser(*this, params);	
}
