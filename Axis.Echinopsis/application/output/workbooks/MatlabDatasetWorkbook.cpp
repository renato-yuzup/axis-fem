#include "stdafx.h"
#include "MatlabDatasetWorkbook.hpp"
#include "application/output/recordsets/MatlabDatasetRecordset.hpp"
#include "application/jobs/AnalysisStepInformation.hpp"
#include "services/locales/LocaleLocator.hpp"
#include "services/locales/Locale.hpp"

namespace aaj = axis::application::jobs;
namespace aaor = axis::application::output::recordsets;
namespace aaow = axis::application::output::workbooks;
namespace asl = axis::services::locales;

aaow::MatlabDatasetWorkbook::MatlabDatasetWorkbook( const axis::String& variableName ) :
variableName_(variableName)
{
  recordset_ = NULL;
}

aaow::MatlabDatasetWorkbook::~MatlabDatasetWorkbook( void )
{
  if (recordset_ != NULL) recordset_->Destroy();
  recordset_ = NULL;
}

void aaow::MatlabDatasetWorkbook::Destroy( void ) const
{
  delete this;
}

bool aaow::MatlabDatasetWorkbook::SupportsAppendOperation( void ) const
{
  return false;
}

axis::String aaow::MatlabDatasetWorkbook::GetFormatIdentifier( void ) const
{
  return _T("m");
}

axis::String aaow::MatlabDatasetWorkbook::GetFormatTitle( void ) const
{
  return _T("MATLAB dataset file");
}

axis::String aaow::MatlabDatasetWorkbook::GetShortDescription( void ) const
{
  return _T("File containing a matrix with analysis data.");
}

bool aaow::MatlabDatasetWorkbook::SupportsNodeRecordset( void ) const
{
  return false;
}

bool aaow::MatlabDatasetWorkbook::SupportsElementRecordset( void ) const
{
  return false;
}

bool aaow::MatlabDatasetWorkbook::SupportsGenericRecordset( void ) const
{
  return true;
}

bool aaow::MatlabDatasetWorkbook::SupportsMainRecordset( void ) const
{
  return false;
}

aaor::ResultRecordset& aaow::MatlabDatasetWorkbook::DoCreateGenericRecordset( aaj::WorkFolder& workFolder )
{
  if (recordset_ == NULL)
  {
    recordset_ = new aaor::MatlabDatasetRecordset(GetWorkbookOutputName(), variableName_, workFolder);
  }
  return *recordset_;
}

void aaow::MatlabDatasetWorkbook::DoAfterOpen( const aaj::AnalysisStepInformation& stepInfo )
{
  asl::LocaleLocator& locator = asl::LocaleLocator::GetLocator();
  const asl::Locale& locale = locator.GetDefaultLocale();
  const asl::DateTimeFacet& facet = locale.GetDataTimeLocale();

  // write header
  axis::String line = _T("%%%");  
  recordset_->RawWriteLine(line);
  line = _T("%%%  JOB TITLE : ") + stepInfo.GetJobTitle(); 
  recordset_->RawWriteLine(line);
  line = _T("%%%  JOB ID    : ") + stepInfo.GetJobId().ToString();
  recordset_->RawWriteLine(line);
  line = _T("%%%  SUBMITED ON : ") + facet.ToLongDateTimeString(stepInfo.GetJobStartTime());
  recordset_->RawWriteLine(line);
  recordset_->RawWriteLine(_T("%%%"));
  recordset_->RawWriteLine(_T("%%% ------"));
  recordset_->RawWriteLine(_T("%%%"));
  line = _T("%%%  THIS IS STEP ") + axis::String::int_parse(stepInfo.GetStepIndex(), 2) + _T(" : ");
  if (stepInfo.GetStepName().empty())
  {
    line += _T("<untitled>");
  }
  else
  {
    line += stepInfo.GetStepName();
  }
  recordset_->RawWriteLine(line);
  recordset_->RawWriteLine(_T(""));
}
