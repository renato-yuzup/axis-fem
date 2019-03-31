#include "TextReportElement6DCollector.hpp"
#include <assert.h>
#include <boost/detail/limits.hpp>
#include "application/output/recordsets/TextReportRecordset.hpp"
#include "domain/algorithms/messages/ModelStateUpdateMessage.hpp"
#include "domain/analyses/NumericalModel.hpp"

namespace aaoc = axis::application::output::collectors;
namespace aaor = axis::application::output::recordsets;
namespace adc = axis::domain::collections;
namespace ada = axis::domain::analyses;
namespace adam = axis::domain::algorithms::messages;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;

namespace {
  static const int idFieldSize = 10;
  static const int fpFieldSize = 24;
  static const axis::String columnSpacing = _T(" ");
  static const axis::String doubleSpacing = _T("  ");
}

aaoc::TextReportElement6DCollector::TextReportElement6DCollector( const axis::String& targetSetName, 
                                                                  const bool *activeDirections )
{
  targetSetName_ = targetSetName;
  for (unsigned int i = 0; i < 6; ++i)
  {
    dofsToPrint_[i] = activeDirections[i];
  }
}

aaoc::TextReportElement6DCollector::~TextReportElement6DCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportElement6DCollector::Collect( const asmm::ResultMessage& message, 
                                                  aaor::ResultRecordset& recordset, 
                                                  const ada::NumericalModel& numericalModel )
{
  aaor::TextReportRecordset& r = static_cast<aaor::TextReportRecordset&>(recordset);

  // initialize statistics variables
  real minVal[6], maxVal[6], avgVal[6], sumVal[6];
  real minAbsVal[6], maxAbsVal[6], avgAbsVal[6], sumAbsVal[6];
  for (int i = 0; i < 6; ++i)
  {
    minVal[i] = std::numeric_limits<real>::max();
    maxVal[i] = std::numeric_limits<real>::min();
    avgVal[i] = 0;
    sumVal[i] = 0;
    minAbsVal[i] = std::numeric_limits<real>::max();
    maxAbsVal[i] = std::numeric_limits<real>::min();
    avgAbsVal[i] = 0;
    sumAbsVal[i] = 0;
  }

  PrintHeader(recordset);

  // write displacement report to file
  const adc::ElementSet& set = numericalModel.GetElementSet(targetSetName_);
  size_type nodeCount = set.Count();
  for (id_type nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx)
  {
    const ade::FiniteElement& element = set.GetByPosition(nodeIdx);

    // write element id
    String s = String::int_parse(element.GetUserId()).align_right(idFieldSize);
    s += columnSpacing + columnSpacing + _T(" ");

    // write dof values
    for (int dirIdx = 0; dirIdx < 6; ++dirIdx)
    {
      if (dofsToPrint_[dirIdx])
      {
        real v = GetDofData(numericalModel, element, dirIdx);
        s += String::double_parse(v).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;

        // collect statistics
        if (v > maxVal[dirIdx]) maxVal[dirIdx] = v;                           // max
        if (v < minVal[dirIdx]) minVal[dirIdx] = v;                           // min
        if (MATH_ABS(v) > maxAbsVal[dirIdx]) maxAbsVal[dirIdx] = MATH_ABS(v); // abs max
        if (MATH_ABS(v) < minAbsVal[dirIdx]) minAbsVal[dirIdx] = MATH_ABS(v); // abs min
        sumVal[dirIdx] += v;                                                  // sum
        sumAbsVal[dirIdx] += MATH_ABS(v);                                     // abs sum
      }
    }
    r.RawWriteLine(s);
  }

  // calculate averages
  if (set.Count() > 0)
  {
    for (int j = 0; j < 6; ++j)
    {
      avgAbsVal[j] = sumAbsVal[j] / (real)set.Count();
      avgVal[j] = sumVal[j] / (real)set.Count();
    }
  }

  // write summary lines
  String summaryMinLine = String(_T("Min :")).align_right(idFieldSize) + doubleSpacing + _T(" "), 
         summaryMaxLine = String(_T("Max :")).align_right(idFieldSize) + doubleSpacing + _T(" "), 
         summaryAvgLine = String(_T("Avg :")).align_right(idFieldSize) + doubleSpacing + _T(" "), 
         summarySumLine = String(_T("Sum :")).align_right(idFieldSize) + doubleSpacing + _T(" "), 
         summaryAbsMinLine = String(_T("Abs min :")).align_right(idFieldSize) + doubleSpacing + _T(" "), 
         summaryAbsMaxLine = String(_T("Abs max :")).align_right(idFieldSize) + doubleSpacing + _T(" "), 
         summaryAbsAvgLine = String(_T("Abs avg :")).align_right(idFieldSize) + doubleSpacing + _T(" "),
         summaryAbsSumLine = String(_T("Abs sum :")).align_right(idFieldSize) + doubleSpacing + _T(" ");
  for (int j = 0; j < 6; ++j)
  {
    if (dofsToPrint_[j])
    {
      summaryMinLine += String::double_parse(minVal[j]).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;
      summaryMaxLine += String::double_parse(maxVal[j]).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;
      summaryAvgLine += String::double_parse(avgVal[j]).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;
      summarySumLine += String::double_parse(sumVal[j]).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;
      summaryAbsMinLine += String::double_parse(minAbsVal[j]).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;
      summaryAbsMaxLine += String::double_parse(maxAbsVal[j]).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;
      summaryAbsAvgLine += String::double_parse(avgAbsVal[j]).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;
      summaryAbsSumLine += String::double_parse(sumAbsVal[j]).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;
    }
  }
  r.RawWriteLine(_T(""));
  r.RawWriteLine(_T(""));
  r.RawWriteLine(summarySumLine);
  r.RawWriteLine(summaryMinLine);
  r.RawWriteLine(summaryMaxLine);
  r.RawWriteLine(summaryAvgLine);
  r.RawWriteLine(_T(""));
  r.RawWriteLine(summaryAbsSumLine);
  r.RawWriteLine(summaryAbsMinLine);
  r.RawWriteLine(summaryAbsMaxLine);
  r.RawWriteLine(summaryAbsAvgLine);
  String separatorLine = String(BuildColumnHeader().size(), '-');
  r.RawWriteLine(separatorLine);
  r.RawWriteLine(_T(""));
  r.RawWriteLine(_T(""));
}

bool aaoc::TextReportElement6DCollector::IsOfInterest( const asmm::ResultMessage& message ) const
{
  return adam::ModelStateUpdateMessage::IsOfKind(message);
}

axis::String aaoc::TextReportElement6DCollector::GetFriendlyDescription( void ) const
{
  String desc;
  int dirCount = 0;
  for (unsigned int i = 0; i < 6; ++i)
  {
    if (dofsToPrint_[i])
    {
      if (!desc.empty()) desc += _T(", ");
      switch (i)
      {
      case 0: desc += _T("XX"); break; 
      case 1: desc += _T("YY"); break; 
      case 2: desc += _T("ZZ"); break; 
      case 3: desc += _T("YZ"); break; 
      case 4: desc += _T("XZ"); break; 
      case 5: desc += _T("XY"); break; 
      default:
        assert(!_T("Unexpected direction!"));
        break;
      }
      dirCount++;
    }
  }
  if (dirCount == 3) desc = _T("All");
  desc +_T(" ") + GetVariableName(dirCount != 1) + _T(" on element set '") + targetSetName_ + _T("'");
  return desc;
}


void aaoc::TextReportElement6DCollector::PrintHeader( aaor::ResultRecordset& recordset )
{
  aaor::TextReportRecordset& r = static_cast<aaor::TextReportRecordset&>(recordset);

  // build headings
  String columnHeaders = BuildColumnHeader();
  String separatorLine = String(columnHeaders.size(), '-');
  String reportTitle = GetVariableName(true).to_upper_case().align_center(columnHeaders.size());

  // write headings  
  r.RawWriteLine(separatorLine);
  r.RawWriteLine(reportTitle);
  r.RawWriteLine(_T(""));
  r.RawWriteLine(_T("  Subject : element set '") + targetSetName_ + _T("'"));
  r.RawWriteLine(separatorLine);
  r.RawWriteLine(columnHeaders);
  r.RawWriteLine(separatorLine);
}

axis::String aaoc::TextReportElement6DCollector::BuildColumnHeader( void )
{
  String columnHeaders = String(_T("element id")).align_center(idFieldSize) + columnSpacing + columnSpacing;	// double spacing
  String v_xx = GetVariableSymbol() + _T("_xx");
  String v_yy = GetVariableSymbol() + _T("_yy");
  String v_zz = GetVariableSymbol() + _T("_zz");
  String v_yz = GetVariableSymbol() + _T("_yz");
  String v_xz = GetVariableSymbol() + _T("_xz");
  String v_xy = GetVariableSymbol() + _T("_xy");

  if (dofsToPrint_[0]) columnHeaders += _T("|") + columnSpacing + v_xx.align_center(fpFieldSize) + columnSpacing;
  if (dofsToPrint_[1]) columnHeaders += _T("|") + columnSpacing + v_yy.align_center(fpFieldSize) + columnSpacing;
  if (dofsToPrint_[2]) columnHeaders += _T("|") + columnSpacing + v_zz.align_center(fpFieldSize) + columnSpacing;
  if (dofsToPrint_[3]) columnHeaders += _T("|") + columnSpacing + v_yz.align_center(fpFieldSize) + columnSpacing;
  if (dofsToPrint_[4]) columnHeaders += _T("|") + columnSpacing + v_xz.align_center(fpFieldSize) + columnSpacing;
  if (dofsToPrint_[5]) columnHeaders += _T("|") + columnSpacing + v_xy.align_center(fpFieldSize) + columnSpacing;
  return columnHeaders;
}

