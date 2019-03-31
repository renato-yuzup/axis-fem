#include "stdafx.h"
#include "HyperworksWorkbookFactory.hpp"
#include "application/output/workbooks/HyperworksWorkbook.hpp"

namespace aafw = axis::application::factories::workbooks;
namespace aaow = axis::application::output::workbooks;
namespace aslse = axis::services::language::syntax::evaluation;


aafw::HyperworksWorkbookFactory::HyperworksWorkbookFactory( void )
{
  // nothing to do here
}

aafw::HyperworksWorkbookFactory::~HyperworksWorkbookFactory( void )
{
  // nothing to do here
}

void aafw::HyperworksWorkbookFactory::Destroy( void ) const
{
  delete this;
}

bool aafw::HyperworksWorkbookFactory::CanBuild( const axis::String& formatName, 
                                                const aslse::ParameterList& formatArguments ) const
{
  return formatName == _T("HYPERWORKS") && formatArguments.IsEmpty();
}

aaow::ResultWorkbook& aafw::HyperworksWorkbookFactory::BuildWorkbook( const axis::String& formatName, 
                                                                      const aslse::ParameterList& formatArguments )
{
  return *new aaow::HyperworksWorkbook();
}
