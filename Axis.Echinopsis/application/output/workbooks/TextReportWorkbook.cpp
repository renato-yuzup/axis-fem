#include "stdafx.h"
#include <assert.h>
#include "TextReportWorkbook.hpp"
#include "application/jobs/AnalysisStepInformation.hpp"
#include "application/output/recordsets/TextReportRecordset.hpp"
#include "services/locales/LocaleLocator.hpp"
#include "services/locales/Locale.hpp"

namespace aaj = axis::application::jobs;
namespace aaow = axis::application::output::workbooks;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace asl = axis::services::locales;

aaow::TextReportWorkbook::TextReportWorkbook( void )
{
  recordset_ = NULL;
}

aaow::TextReportWorkbook::~TextReportWorkbook( void )
{
  // nothing to do here
}

void aaow::TextReportWorkbook::Destroy( void ) const
{
  delete this;
}

bool aaow::TextReportWorkbook::SupportsAppendOperation( void ) const
{
  return false;
}

axis::String aaow::TextReportWorkbook::GetFormatIdentifier( void ) const
{
  return _T("axisreport");
}

axis::String aaow::TextReportWorkbook::GetFormatTitle( void ) const
{
  return _T("Axis Plain-text Report");
}

axis::String aaow::TextReportWorkbook::GetShortDescription( void ) const
{
  return _T("A plain-text report presented in a human-readable format.");
}

bool aaow::TextReportWorkbook::SupportsNodeRecordset( void ) const
{
  return false;
}

bool aaow::TextReportWorkbook::SupportsElementRecordset( void ) const
{
  return false;
}

bool aaow::TextReportWorkbook::SupportsGenericRecordset( void ) const
{
  return true;
}

bool aaow::TextReportWorkbook::SupportsMainRecordset( void ) const
{
  return false;
}

aaor::ResultRecordset& aaow::TextReportWorkbook::DoCreateGenericRecordset( aaj::WorkFolder& workFolder )
{
  if (recordset_ == NULL)
  {
    recordset_ = new aaor::TextReportRecordset(GetWorkbookOutputName(), workFolder);
  }
  return *recordset_;
}

void aaow::TextReportWorkbook::DoAfterOpen( const aaj::AnalysisStepInformation& stepInfo )
{
  asl::LocaleLocator& locator = asl::LocaleLocator::GetLocator();
  const asl::Locale& locale = locator.GetDefaultLocale();
  const asl::DateTimeFacet& facet = locale.GetDataTimeLocale();

  // write header
  axis::String line = _T("************************************************************************************************************************");  
  recordset_->ForcedWriteLine(line);
  line = _T("                                                F I E L D   R E P O R T                                                 "); 
  recordset_->ForcedWriteLine(line);
  recordset_->ForcedWriteLine(_T(""));
  line = _T("  Analysis title : ") + stepInfo.GetJobTitle(); 
  recordset_->ForcedWriteLine(line);
  line = _T("  Run UUID       : ") + stepInfo.GetJobId().ToString();
  recordset_->ForcedWriteLine(line);
  line = _T("  Submitted on   : ") + facet.ToLongDateTimeString(stepInfo.GetJobStartTime());
  line = _T("  Step           : <") + axis::String::int_parse(stepInfo.GetStepIndex()) + _T("> ");
  if (stepInfo.GetStepName().empty())
  {
    line += _T("<untitled>");
  }
  else
  {
    line += stepInfo.GetStepName();
  }
  recordset_->ForcedWriteLine(line);
  recordset_->ForcedWriteLine(_T(""));
  recordset_->ForcedWriteLine(_T(""));
}

void aaow::TextReportWorkbook::DoBeforeClose( void )
{

  recordset_->ForcedWriteLine(_T(""));
  recordset_->ForcedWriteLine(_T(""));
  recordset_->ForcedWriteLine(_T("### END OF REPORT ###"));
  recordset_->ForcedWriteLine(_T(""));
}
