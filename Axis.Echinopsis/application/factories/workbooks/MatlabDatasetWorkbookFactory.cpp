#include "stdafx.h"
#include "MatlabDatasetWorkbookFactory.hpp"
#include "services/language/syntax/evaluation/AtomicValue.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "application/output/workbooks/MatlabDatasetWorkbook.hpp"

namespace aafw = axis::application::factories::workbooks;
namespace aaow = axis::application::output::workbooks;
namespace aslse = axis::services::language::syntax::evaluation;

aafw::MatlabDatasetWorkbookFactory::MatlabDatasetWorkbookFactory( void )
{
  // nothing to do here
}

aafw::MatlabDatasetWorkbookFactory::~MatlabDatasetWorkbookFactory( void )
{
  // nothing to do here
}

void aafw::MatlabDatasetWorkbookFactory::Destroy( void ) const
{
  delete this;
}

bool aafw::MatlabDatasetWorkbookFactory::CanBuild( const axis::String& formatName, 
                                                   const aslse::ParameterList& formatArguments ) const
{
  if (!(formatName == _T("MATLAB_DATASET"))) return false;

  if (formatArguments.IsDeclared(_T("NAME")))
  {
    aslse::ParameterValue& val = formatArguments.GetParameterValue(_T("NAME"));
    if (!val.IsAtomic()) return false;
    aslse::AtomicValue& atomicVal = static_cast<aslse::AtomicValue&>(val);
    if (!(atomicVal.IsId() || atomicVal.IsString())) return false;
    return (formatArguments.Count() == 1);
  }

  return formatArguments.IsEmpty();
}

aaow::ResultWorkbook& aafw::MatlabDatasetWorkbookFactory::BuildWorkbook( 
                                                   const axis::String& formatName, 
                                                   const aslse::ParameterList& formatArguments )
{
  if (!CanBuild(formatName, formatArguments))
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot build the specified format."));
  }

  axis::String varName = _T("data_set");
  if (formatArguments.IsDeclared(_T("NAME")))
  {
    varName = formatArguments.GetParameterValue(_T("NAME")).ToString();
  }

  return *new axis::application::output::workbooks::MatlabDatasetWorkbook(varName);
}
