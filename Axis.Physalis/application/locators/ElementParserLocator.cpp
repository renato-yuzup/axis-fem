#include "ElementParserLocator.hpp"
#include "ElementParserLocator_Pimpl.hpp"
#include "application/parsing/parsers/ElementParser.hpp"
#include "application/parsing/parsers/VoidParser.hpp"
#include "services/language/syntax/evaluation/ParameterValue.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "services/language/syntax/evaluation/NumberValue.hpp"
#include "services/management/ServiceLocator.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/definitions/AxisInputLanguage.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aal = axis::application::locators;
namespace aafp = axis::application::factories::parsers;
namespace aapc = axis::application::parsing::core;
namespace aapps = axis::application::parsing::parsers;
namespace adc = axis::domain::collections;
namespace aslse = axis::services::language::syntax::evaluation;
namespace asmg = axis::services::management;
namespace afd = axis::foundation::definitions;

aal::ElementParserLocator::ElementParserLocator( void )
{
  pimpl_ = new Pimpl();
}

aal::ElementParserLocator::~ElementParserLocator(void)
{
  Pimpl::factory_set::iterator end = pimpl_->factories.end();
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); it != end; ++it)
  {
    aafp::ElementParserFactory& factory = **it;
    factory.Destroy();
  }
	delete pimpl_;
}

aapps::BlockParser& aal::ElementParserLocator::BuildParser( 
    const aapc::SectionDefinition& sectionDefinition, 
    adc::ElementSet& elementCollection ) const
{
	// check if we can build this element
	if (CanBuildElement(sectionDefinition))
	{
    // search for the factory which consumes most of the supplied parameters
    for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); 
         it != pimpl_->factories.end(); ++it)
    {
      if ((*it)->CanBuild(sectionDefinition))
      {
        return (**it).BuildParser(sectionDefinition, elementCollection);
      }
    }
	}
	// can't find a suitable factory
	throw axis::foundation::InvalidOperationException();
}

bool aal::ElementParserLocator::CanBuildElement( 
  const aapc::SectionDefinition& sectionDefinition ) const
{
  // check for any capable factory to build the element
  for (Pimpl::factory_set::iterator it = pimpl_->factories.begin(); 
       it != pimpl_->factories.end(); 
       ++it)
  {
    if ((*it)->CanBuild(sectionDefinition))
    {
      return true;
    }
  }
  // no one can build
  return false;
}

void aal::ElementParserLocator::RegisterFactory( aafp::ElementParserFactory& factory )
{
  // ignore if the factory has already been registered
  if (pimpl_->factories.find(&factory) == pimpl_->factories.end())
  {
    pimpl_->factories.insert(&factory);
  }
}

void aal::ElementParserLocator::UnregisterFactory( aafp::ElementParserFactory& factory )
{
  if (pimpl_->factories.find(&factory) == pimpl_->factories.end())
  {
    throw axis::foundation::ArgumentException();
  }
  pimpl_->factories.erase(&factory);
}

void aal::ElementParserLocator::UnregisterProvider( aafp::BlockProvider& provider )
{
  // we don't allow block nesting
  throw axis::foundation::NotSupportedException();
}

void aal::ElementParserLocator::RegisterProvider( aafp::BlockProvider& provider )
{
  // we don't allow block nesting
  throw axis::foundation::NotSupportedException();
}

bool aal::ElementParserLocator::IsLeaf( void )
{
  return true;
}

bool aal::ElementParserLocator::CanParse( const axis::String& blockName, 
                                          const aslse::ParameterList& params )
{
  // check parameter existence
  if (!params.IsDeclared(afd::AxisInputLanguage::ElementSyntax.SetIdAttributeName) || 
                         params.Count() > 1) /* ==> */ return false;
  // check parameter consistency
  aslse::ParameterValue& v = params.GetParameterValue(
    afd::AxisInputLanguage::ElementSyntax.SetIdAttributeName);
  if (!v.IsAtomic()) return false;
  aslse::AtomicValue& val = (aslse::AtomicValue&)v;
  if (val.IsNumeric())
  {
    aslse::NumberValue& num = (aslse::NumberValue&)val;
    if (!num.IsInteger()) return false;
  }
  if (!val.IsId() && !val.IsString()) return false;
  // check block name
  return (blockName == afd::AxisInputLanguage::ElementSyntax.BlockName);
}

const char * aal::ElementParserLocator::GetFeaturePath( void ) const
{
  return asmg::ServiceLocator::GetFiniteElementLocatorPath();
}

const char * aal::ElementParserLocator::GetFeatureName( void ) const
{
  return "StandardElementParserLocator";
}

aapps::BlockParser& aal::ElementParserLocator::BuildVoidParser( void ) const
{
  return *new aapps::VoidParser(*this);
}

aapps::BlockParser& aal::ElementParserLocator::BuildParser( const axis::String& contextName, 
                                                            const aslse::ParameterList& paramList )
{
  if (!CanParse(contextName, paramList))
  {
    throw axis::foundation::InvalidOperationException();
  }
  return *new aapps::ElementParser(*this, paramList);
}
