#include "NodeSetParserProvider.hpp"
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include "application/parsing/parsers/NodeSetParser.hpp"
#include "foundation/NotSupportedException.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/management/ServiceLocator.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"

namespace aafp  = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg  = axis::services::management;
namespace afd   = axis::foundation::definitions;

aafp::NodeSetParserProvider::NodeSetParserProvider(void)
{
	// nothing to do
}

aafp::NodeSetParserProvider::~NodeSetParserProvider(void)
{
	// nothing to do
}

bool aafp::NodeSetParserProvider::CanParse( const axis::String& blockName, 
                                            const aslse::ParameterList& params )
{
	if (blockName != afd::AxisInputLanguage::NodeSetSyntax.BlockName) return false;
	if (params.Count() != 1) return false;
	if (!params.IsDeclared(afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName)) return false;
	aslse::ParameterValue& val = 
    params.GetParameterValue(afd::AxisInputLanguage::NodeSetSyntax.SetIdAttributeName);
	if (!val.IsAtomic()) return false;
	if (((aslse::AtomicValue&)val).IsNull()) return false;
	if (((aslse::AtomicValue&)val).IsNumeric())
	{
		if (!((aslse::NumberValue&)val).IsInteger()) return false;
	}
	return true;
}

void aafp::NodeSetParserProvider::RegisterProvider( aafp::BlockProvider& provider )
{
	// fail
	throw axis::foundation::NotSupportedException();
}

void aafp::NodeSetParserProvider::UnregisterProvider( aafp::BlockProvider& provider )
{
	// fail
	throw axis::foundation::NotSupportedException();
}

bool aafp::NodeSetParserProvider::IsLeaf( void )
{
	// we won't allow any nested blocks here
	return true;	
}

const char * aafp::NodeSetParserProvider::GetFeaturePath( void ) const
{
	return asmg::ServiceLocator::GetNodeSetInputParserProviderPath();
}

const char * aafp::NodeSetParserProvider::GetFeatureName( void ) const
{
	return "StandardNodeSetParserProvider";
}

aapps::BlockParser& aafp::NodeSetParserProvider::BuildParser( const axis::String& contextName, 
                                                              const aslse::ParameterList& params )
{
	// get params
	return *new aapps::NodeSetParser(*this, params);	
}
