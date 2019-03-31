#include "stdafx.h"
#include "TextReportWorkbookFactory.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "application/output/workbooks/TextReportWorkbook.hpp"

namespace aafw = axis::application::factories::workbooks;
namespace aaow = axis::application::output::workbooks;
namespace aslse = axis::services::language::syntax::evaluation;

aafw::TextReportWorkbookFactory::TextReportWorkbookFactory( void )
{
  // nothing to do here
}

aafw::TextReportWorkbookFactory::~TextReportWorkbookFactory( void )
{
  // nothing to do here
}

void aafw::TextReportWorkbookFactory::Destroy( void ) const
{
  delete this;
}

bool aafw::TextReportWorkbookFactory::CanBuild( const axis::String& formatName, 
                                                const aslse::ParameterList& formatArguments ) const
{
  if (!(formatName == _T("REPORT"))) return false;
  return formatArguments.IsEmpty();
}

aaow::ResultWorkbook& aafw::TextReportWorkbookFactory::BuildWorkbook( 
                                                const axis::String& formatName, 
                                                const aslse::ParameterList& formatArguments )
{
  if (!CanBuild(formatName, formatArguments))
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot build the specified format."));
  }
  return *new aaow::TextReportWorkbook();
}
