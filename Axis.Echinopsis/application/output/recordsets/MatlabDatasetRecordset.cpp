#include "stdafx.h"
#include <assert.h>
#include "MatlabDatasetRecordset.hpp"
#include "application/jobs/WorkFolder.hpp"
#include "domain/analyses/AnalysisInfo.hpp"
#include "domain/analyses/StaticAnalysisInfo.hpp"
#include "domain/analyses/TransientAnalysisInfo.hpp"
#include "domain/analyses/ModalAnalysisInfo.hpp"

namespace aaj = axis::application::jobs;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace asdi = axis::services::diagnostics::information;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;

aaor::MatlabDatasetRecordset::MatlabDatasetRecordset(const axis::String& outputFileName, 
                                                     const axis::String& variableName, 
                                                     aaj::WorkFolder& workFolder) :
variableName_(variableName), writer_(workFolder.GetOrCreateWorkFile(outputFileName))
{
  // nothing to do here
}

aaor::MatlabDatasetRecordset::~MatlabDatasetRecordset(void)
{
  if (writer_.IsOpen())
  {
    writer_.Close();
  }
  writer_.Destroy();
}

void aaor::MatlabDatasetRecordset::Destroy( void ) const
{
  delete this;
}

void aaor::MatlabDatasetRecordset::OpenRecordset( const axis::String& )
{
  writer_.Open();
}

void aaor::MatlabDatasetRecordset::CloseRecordset( void )
{
  writer_.Close();
}

void aaor::MatlabDatasetRecordset::BeginAnalysisStep( const axis::String&, int, 
                                                      const asdi::SolverCapabilities& )
{
  RawWriteLine(variableName_ + _T(" = ["));
}

void aaor::MatlabDatasetRecordset::EndAnalysisStep( void )
{
  pendingLine_.clear();
  RawWriteLine(_T(""));
  RawWriteLine(_T("];"));
  RawWriteLine(_T(""));
}

void aaor::MatlabDatasetRecordset::BeginSnapshot( const ada::AnalysisInfo& analysisInfo )
{
  if (!pendingLine_.empty())
  {
    RawWriteLine(pendingLine_);
    pendingLine_.clear();
  }
  axis::String line;
  switch (analysisInfo.GetAnalysisType())
  {
  case ada::AnalysisInfo::TransientAnalysis:
    {
      const ada::TransientAnalysisInfo& dynamicInfo = 
          static_cast<const ada::TransientAnalysisInfo&>(analysisInfo);
      line = _T("    ") + axis::String::double_parse(dynamicInfo.GetCurrentAnalysisTime());
      break;
    }
  case ada::AnalysisInfo::ModalAnalysis:
    {
      const ada::ModalAnalysisInfo& modalInfo = static_cast<const ada::ModalAnalysisInfo&>(analysisInfo);
      line = _T("    ") + axis::String::int_parse(modalInfo.GetCurrentModeIndex()) +
             _T("    ") + axis::String::double_parse(modalInfo.GetCurrentFrequency());
      break;
    }
  default:
    line = _T("    ");
  }
  RawWrite(line);
}

void aaor::MatlabDatasetRecordset::EndSnapshot( const ada::AnalysisInfo& analysisInfo )
{
  pendingLine_ = _T(" ; ");
}

bool aaor::MatlabDatasetRecordset::IsInitialized( void ) const
{
  return true;
}

bool aaor::MatlabDatasetRecordset::IsReady( void ) const
{
  return writer_.IsOpen();
}

void aaor::MatlabDatasetRecordset::WriteData( int data )
{
  axis::String s = axis::String::int_parse(data).align_right(16);
  RawWrite(_T("    ") + s);
}

void aaor::MatlabDatasetRecordset::WriteData( real data )
{
  axis::String s = axis::String::double_parse(data);
  RawWrite(_T("    ") + s);
}

void aaor::MatlabDatasetRecordset::WriteData( bool data )
{
  if (data)
  {
    RawWrite(_T("     true"));
  }
  else
  {
    RawWrite(_T("    false"));
  }
}

void aaor::MatlabDatasetRecordset::WriteData( const axis::String& literal )
{
  RawWrite(_T("    ") + literal);
}

void aaor::MatlabDatasetRecordset::WriteData( const afb::DenseMatrix& data )
{
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; rowIdx++)
  {
    for (size_type colIdx = 0; colIdx < colCount; colIdx++)
    {
      WriteData(data(rowIdx, colIdx));
    }
  }
}

void aaor::MatlabDatasetRecordset::WriteData( const afb::SymmetricMatrix& data )
{
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; rowIdx++)
  {
    for (size_type colIdx = 0; colIdx < colCount; colIdx++)
    {
      WriteData(data(rowIdx, colIdx));
    }
  }
}

void aaor::MatlabDatasetRecordset::WriteData( const afb::LowerTriangularMatrix& data )
{
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; rowIdx++)
  {
    for (size_type colIdx = 0; colIdx < colCount; colIdx++)
    {
      WriteData(data(rowIdx, colIdx));
    }
  }
}

void aaor::MatlabDatasetRecordset::WriteData( const afb::UpperTriangularMatrix& data )
{
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; rowIdx++)
  {
    for (size_type colIdx = 0; colIdx < colCount; colIdx++)
    {
      WriteData(data(rowIdx, colIdx));
    }
  }
}

void aaor::MatlabDatasetRecordset::WriteData( const afb::ColumnVector& data )
{
  size_type count = data.Length();
  for (size_type idx = 0; idx < count; idx++)
  {
    WriteData(data(idx));
  }
}

void aaor::MatlabDatasetRecordset::WriteData( const afb::RowVector& data )
{
  size_type count = data.Length();
  for (size_type idx = 0; idx < count; idx++)
  {
    WriteData(data(idx));
  }
}

void aaor::MatlabDatasetRecordset::RawWriteLine( const axis::String& line )
{
  writer_.WriteLine(line);
}

void aaor::MatlabDatasetRecordset::RawWrite( const axis::String& line )
{
  writer_.Write(line);
}
