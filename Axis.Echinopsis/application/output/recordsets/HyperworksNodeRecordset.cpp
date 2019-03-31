#include "stdafx.h"
#include "HyperworksNodeRecordset.hpp"
#include "application/jobs/WorkFolder.hpp"
#include "domain/analyses/AnalysisInfo.hpp"
#include "domain/analyses/StaticAnalysisInfo.hpp"
#include "domain/analyses/TransientAnalysisInfo.hpp"
#include "domain/analyses/ModalAnalysisInfo.hpp"
#include "domain/elements/Node.hpp"
#include "services/io/FileSystem.hpp"

#define INTERNAL_BUFFER_LENGTH     1024*1024    // 1 MB

namespace aaj = axis::application::jobs;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asdi = axis::services::diagnostics::information;
namespace asio = axis::services::io;
namespace asmm = axis::services::messaging;
namespace afb = axis::foundation::blas;

namespace {
  inline void WriteStaticAnalysisHeader( const ada::StaticAnalysisInfo&, 
                                         asio::StreamWriter& ) 
  {
    // nothing to do, simply ignore 
  }
  inline void WriteTransientAnalysisHeader( axis::String& buffer,
    const ada::TransientAnalysisInfo& info, asio::StreamWriter& writer ) 
  {
    axis::String& s = buffer;
    s = _T("$TIME = ");
    s += axis::String::double_parse(info.GetCurrentAnalysisTime());
    writer.WriteLine(s);
  }
  inline void WriteModalAnalysisHeader( axis::String& buffer,
                                        const ada::ModalAnalysisInfo& info, 
                                        asio::StreamWriter& writer ) 
  {
    buffer = _T("$FREQUENCY = ");
    buffer += axis::String::double_parse(info.GetCurrentFrequency());
    writer.WriteLine(buffer);
    buffer = _T("$MODE = ");
    buffer += axis::String::int_parse(info.GetCurrentModeIndex());
    writer.WriteLine(buffer);
  }
  inline void WriteString(const axis::String& s, asio::StreamWriter& writer)
  {
    const static axis::String space = _T("    ");
    writer.Write(space);
    writer.Write(s);
  }
}

aaor::HyperworksNodeRecordset::HyperworksNodeRecordset( void ) :
  internalBuffer_(INTERNAL_BUFFER_LENGTH, ' ')
{
  writer_ = NULL;
}

aaor::HyperworksNodeRecordset::~HyperworksNodeRecordset( void )
{
  if (writer_->IsOpen())
  {
    writer_->Close();
  }
  writer_ = NULL;
}

void aaor::HyperworksNodeRecordset::Destroy( void ) const
{
  delete this;
}

void aaor::HyperworksNodeRecordset::Init( aaj::WorkFolder& workFolder )
{
  writer_ = &workFolder.CreateTempFile(_T("hwascii_node"), _T("hwtemp"));
}

void aaor::HyperworksNodeRecordset::OpenRecordset( const axis::String& entitySet )
{
  writer_->Open(asio::StreamWriter::kOverwrite);
}

void aaor::HyperworksNodeRecordset::CloseRecordset( void )
{
  writer_->Close();
}

void aaor::HyperworksNodeRecordset::BeginCreateField( void )
{
  resultTypeStatement_.clear();
}

void aaor::HyperworksNodeRecordset::CreateField( const axis::String& fieldName, FieldType )
{
  WriteFieldName(fieldName);
}

void aaor::HyperworksNodeRecordset::CreateMatrixField( const axis::String& fieldName, int, int )
{
  WriteFieldName(fieldName);
}

void aaor::HyperworksNodeRecordset::CreateVectorField( const axis::String& fieldName, int )
{
  WriteFieldName(fieldName);
}

void aaor::HyperworksNodeRecordset::EndCreateField( void )
{
  resultTypeStatement_ = _T("$RESULT_TYPE = ") + resultTypeStatement_;
}

void aaor::HyperworksNodeRecordset::BeginAnalysisStep( const axis::String& stepName, int stepIndex, 
                                                       const asdi::SolverCapabilities& solverCaps )
{
  internalBuffer_ = _T("$BINDING = NODE");
  writer_->WriteLine(internalBuffer_);
  internalBuffer_ = _T("$COLUMN_INFO = ENTITY_ID");
  writer_->WriteLine(internalBuffer_);
  writer_->WriteLine(resultTypeStatement_);
}

void aaor::HyperworksNodeRecordset::EndAnalysisStep( void )
{
  writer_->WriteLine();
}

void aaor::HyperworksNodeRecordset::BeginSnapshot( const ada::AnalysisInfo& analysisInfo )
{
  switch (analysisInfo.GetAnalysisType())
  {
  case ada::AnalysisInfo::StaticAnalysis:
    {
      const ada::StaticAnalysisInfo& info = static_cast<const ada::StaticAnalysisInfo&>(analysisInfo);
      WriteStaticAnalysisHeader(info, *writer_);
    }
    break;
  case ada::AnalysisInfo::TransientAnalysis:
    {
      const ada::TransientAnalysisInfo& info = static_cast<const ada::TransientAnalysisInfo&>(analysisInfo);
      WriteTransientAnalysisHeader(internalBuffer_, info, *writer_);
    }
    break;
  case ada::AnalysisInfo::ModalAnalysis:
    {
      const ada::ModalAnalysisInfo& info = static_cast<const ada::ModalAnalysisInfo&>(analysisInfo);
      WriteModalAnalysisHeader(internalBuffer_, info, *writer_);
    }
    break;
  default:
    break;
  }
}

void aaor::HyperworksNodeRecordset::BeginNodeRecord( const asmm::ResultMessage&, const ade::Node& node )
{
  writer_->Write(axis::String::int_parse(node.GetUserId()).align_left(12));
}

void aaor::HyperworksNodeRecordset::EndNodeRecord( const asmm::ResultMessage&, const ade::Node& )
{
  writer_->WriteLine();
}

bool aaor::HyperworksNodeRecordset::IsInitialized( void ) const
{
  return writer_ != NULL;
}

bool aaor::HyperworksNodeRecordset::IsReady( void ) const
{
  return writer_->IsOpen();
}

void aaor::HyperworksNodeRecordset::WriteData( int data )
{
  WriteString(axis::String::int_parse(data), *writer_);
}

void aaor::HyperworksNodeRecordset::WriteData( real data )
{
  WriteString(axis::String::double_parse(data), *writer_);
}

void aaor::HyperworksNodeRecordset::WriteData( bool data )
{
  WriteString(data? _T("true") : _T("false"), *writer_);
}

void aaor::HyperworksNodeRecordset::WriteData( const axis::String& literal )
{
  WriteString(literal, *writer_);
}

void aaor::HyperworksNodeRecordset::WriteData( const afb::DenseMatrix& data )
{
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; ++rowIdx)
  {
    for (size_type colIdx = 0; colIdx < colCount; ++colIdx)
    {
      WriteString(axis::String::double_parse(data(rowIdx, colIdx)), *writer_);
    }
  }
}

void aaor::HyperworksNodeRecordset::WriteData( const afb::SymmetricMatrix& data )
{
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; ++rowIdx)
  {
    for (size_type colIdx = 0; colIdx < colCount; ++colIdx)
    {
      WriteString(axis::String::double_parse(data(rowIdx, colIdx)), *writer_);
    }
  }
}

void aaor::HyperworksNodeRecordset::WriteData( const afb::LowerTriangularMatrix& data )
{
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; ++rowIdx)
  {
    for (size_type colIdx = 0; colIdx < colCount; ++colIdx)
    {
      WriteString(axis::String::double_parse(data(rowIdx, colIdx)), *writer_);
    }
  }
}

void aaor::HyperworksNodeRecordset::WriteData( const afb::UpperTriangularMatrix& data )
{
  size_type rowCount = data.Rows();
  size_type colCount = data.Columns();
  for (size_type rowIdx = 0; rowIdx < rowCount; ++rowIdx)
  {
    for (size_type colIdx = 0; colIdx < colCount; ++colIdx)
    {
      WriteString(axis::String::double_parse(data(rowIdx, colIdx)), *writer_);
    }
  }
}

void aaor::HyperworksNodeRecordset::WriteData( const afb::ColumnVector& data )
{
  size_type count = data.Length();
  for (size_type idx = 0; idx < count; ++idx)
  {
    WriteString(axis::String::double_parse(data(idx)), *writer_);
  }
}

void aaor::HyperworksNodeRecordset::WriteData( const afb::RowVector& data )
{
  size_type count = data.Length();
  for (size_type idx = 0; idx < count; ++idx)
  {
    WriteString(axis::String::double_parse(data(idx)), *writer_);
  }
}

axis::String aaor::HyperworksNodeRecordset::GetTempFileLocation( void ) const
{
  return asio::FileSystem::GetFileTitle(writer_->GetStreamPath()) + _T(".hwtemp");
}

void aaor::HyperworksNodeRecordset::WriteFieldName( const axis::String& fieldName )
{
  if (!resultTypeStatement_.empty())
  {
    resultTypeStatement_ += _T(", ");
  }
  resultTypeStatement_ += fieldName;
}
