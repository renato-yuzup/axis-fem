#include "stdafx.h"
#include <assert.h>
#include "TextReportRecordset.hpp"
#include "application/jobs/WorkFolder.hpp"
#include "domain/analyses/AnalysisInfo.hpp"
#include "domain/analyses/TransientAnalysisInfo.hpp"

namespace aaj = axis::application::jobs;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace asdi = axis::services::diagnostics::information;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;

aaor::TextReportRecordset::TextReportRecordset( const axis::String& outputFileName, 
                                                aaj::WorkFolder& workFolder ) :
writer_(workFolder.GetOrCreateWorkFile(outputFileName))
{
  sleeping_ = true;
}

aaor::TextReportRecordset::~TextReportRecordset( void )
{
  if (writer_.IsOpen())
  {
    writer_.Close();
  }
  writer_.Destroy();
}

void aaor::TextReportRecordset::Destroy( void ) const
{
  delete this;
}

void aaor::TextReportRecordset::OpenRecordset( const axis::String& entitySet )
{
  writer_.Open();
}

void aaor::TextReportRecordset::CloseRecordset( void )
{
  writer_.Close();
}

void aaor::TextReportRecordset::BeginSnapshot( const ada::AnalysisInfo& analysisInfo )
{
  switch (analysisInfo.GetAnalysisType())
  {
  case ada::AnalysisInfo::StaticAnalysis:
  case ada::AnalysisInfo::ModalAnalysis:
    sleeping_ = false;
    break;
  case ada::AnalysisInfo::TransientAnalysis:
    {
      const ada::TransientAnalysisInfo& dynInfo = 
          static_cast<const ada::TransientAnalysisInfo&>(analysisInfo);
      if (dynInfo.GetCurrentAnalysisTime() >= dynInfo.GetEndTime())
      {
        sleeping_ = false;
      }
      break;
    }
  default:
    assert(!_T("Unknown analysis type."));
    break;
  }
}

bool aaor::TextReportRecordset::IsInitialized( void ) const
{
  return true;
}

bool aaor::TextReportRecordset::IsReady( void ) const
{
  return writer_.IsOpen();
}

void aaor::TextReportRecordset::WriteData( int data )
{
  if (sleeping_) return;
  RawWrite(String::int_parse(data));
}

void aaor::TextReportRecordset::WriteData( real data )
{
  if (sleeping_) return;
  RawWrite(String::double_parse(data));
}

void aaor::TextReportRecordset::WriteData( bool data )
{
  if (sleeping_) return;
  RawWrite(data? _T("true") : _T("false"));
}
void aaor::TextReportRecordset::WriteData( const axis::String& literal )
{
  if (sleeping_) return;
  RawWrite(literal);
}

void aaor::TextReportRecordset::WriteData( const afb::DenseMatrix& data )
{
  if (sleeping_) return;
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; ++rowIdx)
  {
    for (size_type colIdx = 0; colIdx < colCount; ++colIdx)
    {
      WriteData(data(rowIdx, colIdx));
    }
  }
}

void aaor::TextReportRecordset::WriteData( const afb::SymmetricMatrix& data )
{
  if (sleeping_) return;
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; ++rowIdx)
  {
    for (size_type colIdx = 0; colIdx < colCount; ++colIdx)
    {
      WriteData(data(rowIdx, colIdx));
    }
  }
}

void aaor::TextReportRecordset::WriteData( const afb::LowerTriangularMatrix& data )
{
  if (sleeping_) return;
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; ++rowIdx)
  {
    for (size_type colIdx = 0; colIdx < colCount; ++colIdx)
    {
      WriteData(data(rowIdx, colIdx));
    }
  }
}

void aaor::TextReportRecordset::WriteData( const afb::UpperTriangularMatrix& data )
{
  if (sleeping_) return;
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; ++rowIdx)
  {
    for (size_type colIdx = 0; colIdx < colCount; ++colIdx)
    {
      WriteData(data(rowIdx, colIdx));
    }
  }
}

void aaor::TextReportRecordset::WriteData( const afb::ColumnVector& data )
{
  if (sleeping_) return;
  size_type len = data.Length();
  for (size_type idx = 0; idx < len; ++idx)
  {
    WriteData(data(idx));
  }
}

void aaor::TextReportRecordset::WriteData( const afb::RowVector& data )
{
  if (sleeping_) return;
  size_type len = data.Length();
  for (size_type idx = 0; idx < len; ++idx)
  {
    WriteData(data(idx));
  }
}

void aaor::TextReportRecordset::EndGenericRecord( const asmm::ResultMessage&, 
                                                  const ada::NumericalModel& )
{
  RawWriteLine(_T(""));
  RawWriteLine(_T(""));
}

void aaor::TextReportRecordset::RawWriteLine( const axis::String& line )
{
  if (sleeping_) return;
  writer_.WriteLine(line);
}

void aaor::TextReportRecordset::RawWrite( const axis::String& line )
{
  if (sleeping_) return;
  writer_.Write(line);
}

void axis::application::output::recordsets::TextReportRecordset::ForcedWriteLine( const axis::String& line )
{
  writer_.WriteLine(line);
}

void axis::application::output::recordsets::TextReportRecordset::ForcedWrite( const axis::String& line )
{
  writer_.Write(line);
}
