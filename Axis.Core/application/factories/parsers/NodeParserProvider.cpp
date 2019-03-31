#include "NodeParserProvider.hpp"
#include "application/parsing/parsers/NodeParser.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "services/management/ServiceLocator.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"

namespace aafe = axis::application::factories::elements;
namespace aafp = axis::application::factories::parsers;
namespace aapps = axis::application::parsing::parsers;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;
namespace afd = axis::foundation::definitions;

aafp::NodeParserProvider::NodeParserProvider(void)
{
}

aafp::NodeParserProvider::~NodeParserProvider(void)
{
}

bool aafp::NodeParserProvider::CanParse( const axis::String& blockName, 
                                         const aslse::ParameterList& params )
{
	if (blockName != afd::AxisInputLanguage::NodeSyntax.BlockName) return false;
	
	if (params.Count() > 0)
	{
		if (params.Count() != 1) return false;
		if (!params.IsDeclared(afd::AxisInputLanguage::NodeSyntax.SetIdAttributeName)) return false;
		aslse::ParameterValue& val = 
      params.GetParameterValue(afd::AxisInputLanguage::NodeSyntax.SetIdAttributeName);
		if (!val.IsAtomic()) return false;
    if (((aslse::AtomicValue&)val).IsNull()) return false;
    if (((aslse::AtomicValue&)val).IsNumeric())
		{
			if (!((aslse::NumberValue&)val).IsInteger()) return false;
		}
		
	}
	return true;
}

const char * aafp::NodeParserProvider::GetFeaturePath( void ) const
{
	return asmg::ServiceLocator::GetNodeInputParserProviderPath();
}

const char * aafp::NodeParserProvider::GetFeatureName( void ) const
{
	return "StandardNodeParserProvider";
}

void aafp::NodeParserProvider::DoOnPostProcessRegistration( asmg::GlobalProviderCatalog& manager )
{
	// we registered successfully; get the standard node factory
	asmg::Provider& provider = manager.GetProvider(asmg::ServiceLocator::GetNodeFactoryPath());
	_nodeFactory = static_cast<aafe::NodeFactory *>(&provider);
}

void aafp::NodeParserProvider::DoOnUnload( asmg::GlobalProviderCatalog& manager )
{
	_nodeFactory = NULL;
}

aapps::BlockParser& aafp::NodeParserProvider::BuildParser( const axis::String& contextName, 
                                                           const aslse::ParameterList& params )
{
	return *new aapps::NodeParser(*this, *_nodeFactory, params);
}
