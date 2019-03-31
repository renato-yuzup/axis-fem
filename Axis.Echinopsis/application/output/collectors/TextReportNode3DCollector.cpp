#include "TextReportNode3DCollector.hpp"
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

aaoc::TextReportNode3DCollector::TextReportNode3DCollector( const axis::String& targetSetName, 
                                                            const bool *activeDirections )
{
  targetSetName_ = targetSetName;
  for (unsigned int i = 0; i < 3; ++i)
  {
    dofsToPrint_[i] = activeDirections[i];
  }
}

aaoc::TextReportNode3DCollector::~TextReportNode3DCollector( void )
{
  // nothing to do here
}

void aaoc::TextReportNode3DCollector::Collect( const asmm::ResultMessage& message, 
                                               aaor::ResultRecordset& recordset, 
                                               const ada::NumericalModel& numericalModel )
{
  aaor::TextReportRecordset& r = static_cast<aaor::TextReportRecordset&>(recordset);

  // initialize statistics variables
  real minVal[3], maxVal[3], avgVal[3], sumVal[3];
  real minAbsVal[3], maxAbsVal[3], avgAbsVal[3], sumAbsVal[3];
  for (int i = 0; i < 3; ++i)
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
  const adc::NodeSet& set = numericalModel.GetNodeSet(targetSetName_);
  size_type nodeCount = set.Count();
  for (id_type nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx)
  {
    const ade::Node& node = set.GetByPosition(nodeIdx);

    // write node id
    String s = String::int_parse(node.GetUserId()).align_right(idFieldSize);
    s += columnSpacing + columnSpacing + _T(" ");

    // write dof values
    for (int dofIdx = 0; dofIdx < 3; ++dofIdx)
    {
      if (dofsToPrint_[dofIdx])
      {
        real v = GetDofData(numericalModel, node, dofIdx);
        s += String::double_parse(v).align_right(fpFieldSize + columnSpacing.size() + 1) + columnSpacing;

        // collect statistics
        if (v > maxVal[dofIdx]) maxVal[dofIdx] = v;                           // max
        if (v < minVal[dofIdx]) minVal[dofIdx] = v;                           // min
        if (MATH_ABS(v) > maxAbsVal[dofIdx]) maxAbsVal[dofIdx] = MATH_ABS(v); // abs max
        if (MATH_ABS(v) < minAbsVal[dofIdx]) minAbsVal[dofIdx] = MATH_ABS(v); // abs min
        sumVal[dofIdx] += v;                                                  // sum
        sumAbsVal[dofIdx] += MATH_ABS(v);                                     // abs sum
      }
    }
    r.RawWriteLine(s);
  }

  // calculate averages
  if (set.Count() > 0)
  {
    for (int j = 0; j < 3; ++j)
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
  for (int j = 0; j < 3; ++j)
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

bool aaoc::TextReportNode3DCollector::IsOfInterest( const asmm::ResultMessage& message ) const
{
  return adam::ModelStateUpdateMessage::IsOfKind(message);
}

axis::String aaoc::TextReportNode3DCollector::GetFriendlyDescription( void ) const
{
  String desc;
  int dirCount = 0;
  for (unsigned int i = 0; i < 3; ++i)
  {
    if (dofsToPrint_[i])
    {
      if (!desc.empty()) desc += _T(", ");
      switch (i)
      {
      case 0: desc += _T("X"); break; 
      case 1: desc += _T("Y"); break; 
      case 2: desc += _T("Z"); break; 
      default:
        assert(!_T("Unexpected direction!"));
        break;
      }
      dirCount++;
    }
  }
  if (dirCount == 3) desc = _T("All");
  desc +_T(" ") + GetVariableName(dirCount != 1) + _T(" on node set '") + targetSetName_ + _T("'");
  return desc;
}


void aaoc::TextReportNode3DCollector::PrintHeader( aaor::ResultRecordset& recordset )
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
  r.RawWriteLine(_T("  Subject : node set '") + targetSetName_ + _T("'"));
  r.RawWriteLine(separatorLine);
  r.RawWriteLine(columnHeaders);
  r.RawWriteLine(separatorLine);
}

axis::String aaoc::TextReportNode3DCollector::BuildColumnHeader( void )
{
  String columnHeaders = String(_T("node id")).align_center(idFieldSize) + columnSpacing + columnSpacing;	// double spacing
  String v_x = GetVariableSymbol() + _T("_x");
  String v_y = GetVariableSymbol() + _T("_y");
  String v_z = GetVariableSymbol() + _T("_z");

  if (dofsToPrint_[0]) columnHeaders += _T("|") + columnSpacing + v_x.align_center(fpFieldSize) + columnSpacing;
  if (dofsToPrint_[1]) columnHeaders += _T("|") + columnSpacing + v_y.align_center(fpFieldSize) + columnSpacing;
  if (dofsToPrint_[2]) columnHeaders += _T("|") + columnSpacing + v_z.align_center(fpFieldSize) + columnSpacing;
  return columnHeaders;
}



