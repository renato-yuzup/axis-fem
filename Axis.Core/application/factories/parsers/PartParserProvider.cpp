#include "PartParserProvider.hpp"
#include "services/language/syntax/evaluation/ArrayValue.hpp"
#include "application/parsing/parsers/PartParser.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "services/management/ServiceLocator.hpp"

namespace aafp = axis::application::factories::parsers;
namespace aal = axis::application::locators;
namespace aapps = axis::application::parsing::parsers;
namespace asmg = axis::services::management;
namespace aslse = axis::services::language::syntax::evaluation;
namespace afd = axis::foundation::definitions;

aafp::PartParserProvider::PartParserProvider(void)
{
	_elementProvider = NULL;
	_modelProvider = NULL;
}

aafp::PartParserProvider::~PartParserProvider(void)
{
	// nothing to do here
}

bool aafp::PartParserProvider::CanParse( const axis::String& blockName, const aslse::ParameterList& params )
{
	// we can only parse the PARTS block
	if (blockName.compare(afd::AxisInputLanguage::PartSyntax.BlockName) != 0)
	{
		return false;
	}

	// check for the element description and type parameters
	if (!(params.Count() == 2 && 
      params.IsDeclared(afd::AxisInputLanguage::PartSyntax.ElementDescriptionParameterName) &&
		  params.IsDeclared(afd::AxisInputLanguage::PartSyntax.ElementTypeParameterName)))
	{
		return false;
	}

	// everything went accordingly
	return true;
}

aapps::BlockParser& aafp::PartParserProvider::BuildParser( const axis::String& contextName, 
                                                           const aslse::ParameterList& params )
{
	axis::String elementType;
	aapps::BlockParser *parser;
	// check if we can really parse this
	if (!CanParse(contextName, params))
	{
		throw axis::foundation::InvalidOperationException();
	}
	// create part parser for elements construction annotation
	parser = new aapps::PartParser(*this, *_elementProvider, *_modelProvider, params);
	return *parser;
}

void aafp::PartParserProvider::DoOnPostProcessRegistration( asmg::GlobalProviderCatalog& rootManager )
{
	// get our auxiliary provider
  _elementProvider = &static_cast<aal::ElementParserLocator&>(
    rootManager.GetProvider(asmg::ServiceLocator::GetFiniteElementLocatorPath()));
  _modelProvider = &static_cast<aal::MaterialFactoryLocator&>(
    rootManager.GetProvider(asmg::ServiceLocator::GetMaterialFactoryLocatorPath()));
}

const char * aafp::PartParserProvider::GetFeaturePath( void ) const
{
	return asmg::ServiceLocator::GetPartInputParserProviderPath();
}

const char * aafp::PartParserProvider::GetFeatureName( void ) const
{
	return "StandardPartAssemblerProvider";
}
